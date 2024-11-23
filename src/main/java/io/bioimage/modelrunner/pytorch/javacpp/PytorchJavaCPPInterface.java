/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Pytorch. This project uses Pytorch with thanks to JavaCPP
 * %%
 * Copyright (C) 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.pytorch.javacpp;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.IValue;
import org.bytedeco.pytorch.IValueVector;
import org.bytedeco.pytorch.JitModule;
import org.bytedeco.pytorch.TensorVector;

import com.google.gson.Gson;

import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.JavaCPPTensorBuilder;
import io.bioimage.modelrunner.pytorch.javacpp.shm.ShmBuilder;
import io.bioimage.modelrunner.pytorch.javacpp.shm.TensorBuilder;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

/**
 * This class implements an interface that allows the main plugin to interact in
 * an agnostic way with the Deep Learning model and tensors to make inference.
 * This implementation adds Pytorch support to the main program using the JavaCPP
 * library to load the corresponding native bindings.
 * 
 * Class to that communicates with the dl-model runner, 
 * @see <a href="https://github.com/bioimage-io/model-runner-java">dlmodelrunner</a>
 * to execute Pytorch models.
 * This class implements the interface {@link DeepLearningEngineInterface} to get the 
 * agnostic {@link io.bioimage.modelrunner.tensor.Tensor}, convert them into 
 * {@link org.bytedeco.pytorch.Tensor}, execute a Pytorch Deep Learning model on them and
 * convert the results back to {@link io.bioimage.modelrunner.tensor.Tensor} to send them 
 * to the main program in an agnostic manner.
 * 
 * {@link ImgLib2Builder}. Creates ImgLib2 images for the backend
 *  of {@link io.bioimage.modelrunner.tensor.Tensor} from {@link org.bytedeco.pytorch.Tensor}
 * {@link JavaCPPTensorBuilder}. Converts {@link io.bioimage.modelrunner.tensor.Tensor} 
 *  into {@link org.bytedeco.pytorch.Tensor}
 *  
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class PytorchJavaCPPInterface implements DeepLearningEngineInterface
{

	/**
	 * The Pytorch torchscript model loaded with JavaCpp
	 */
	private JitModule model;
	
	private String modelFolder;
	
	private String modelSource;
	
	private boolean interprocessing = true;
    /**
     * Process where the model is being loaded and executed
     */
    Service runner;

	private List<SharedMemoryArray> shmaInputList = new ArrayList<SharedMemoryArray>();
	private List<SharedMemoryArray> shmaOutputList = new ArrayList<SharedMemoryArray>();
	
	private List<String> shmaNamesList = new ArrayList<String>();

	private static final String NAME_KEY = "name";
	private static final String SHAPE_KEY = "shape";
	private static final String DTYPE_KEY = "dtype";
	private static final String IS_INPUT_KEY = "isInput";
	private static final String MEM_NAME_KEY = "memoryName";
	/**
	 * Name without vesion of the JAR created for this library
	 */
	private static final String JAR_FILE_NAME = "dl-modelrunner-pytorch-javacpp";

	/**
	 * Constructor for the interface. It is going to be called from the 
	 * dlmodel-runner.
	 * This one tries to use mkl to be faster
	 */
    public PytorchJavaCPPInterface() throws IOException, URISyntaxException
    {
		this(true);
    }
	
    /**
     * Private constructor that can only be launched from the class to create a separate
     * process to avoid the conflicts that occur in the same process between Pytorch1 and 2
     * @param doInterprocessing
     * 	whether to do interprocessing or not
     */
    public PytorchJavaCPPInterface(boolean doInterprocessing) throws IOException, URISyntaxException
    {
		interprocessing = doInterprocessing;
		if (this.interprocessing) {
			runner = getRunner();
			runner.debug((text) -> System.err.println(text));
		}
    }
    
    private Service getRunner() throws IOException, URISyntaxException {
		List<String> args = getProcessCommandsWithoutArgs();
		String[] argArr = new String[args.size()];
		args.toArray(argArr);

		return new Service(new File("."), argArr);
    }

	/**
	 * {@inheritDoc}
	 * 
     * Load a Pytorch torchscript model using JavaCpp. 
	 */
	@Override
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
		this.modelFolder = modelFolder;
		this.modelSource = modelSource;
		if (interprocessing) {
			try {
				launchModelLoadOnProcess();
			} catch (IOException | InterruptedException e) {
				throw new LoadModelException(Types.stackTrace(e));
			}
			return;
		}

    	try {
	    	model = org.bytedeco.pytorch.global.torch.load(modelSource);
			model.eval();
    	} catch (Exception ex) {
    		throw new LoadModelException(Types.stackTrace(ex));
    	}
	}
	
	private void launchModelLoadOnProcess() throws IOException, InterruptedException {
		HashMap<String, Object> args = new HashMap<String, Object>();
		args.put("modelFolder", this.modelFolder);
		args.put("modelSource", this.modelSource);
		Task task = runner.task("loadModel", args);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException();
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException();
		else if (task.status == TaskStatus.CRASHED) {
			this.runner.close();
			runner = null;
			throw new RuntimeException();
		}
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Run a Pytorch model using JavaCpp on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 * 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		if (interprocessing) {
			runInterprocessing(inputTensors, outputTensors);
			return;
		}
		IValueVector inputsVector = new IValueVector();
		List<String> inputListNames = new ArrayList<String>();
        for (Tensor<T> tt : inputTensors) {
        	inputListNames.add(tt.getName());
        	inputsVector.put(new IValue(JavaCPPTensorBuilder.build(tt)));
        }
        // Run model
		model.eval();
        IValue output = model.forward(inputsVector);
        TensorVector outputTensorVector = null;
        if (output.isTensorList()) {
        	outputTensorVector = output.toTensorVector();
        } else {
        	outputTensorVector = new TensorVector();
        	outputTensorVector.put(output.toTensor());
        }
		// Fill the agnostic output tensors list with data from the inference result
		fillOutputTensors(outputTensorVector, outputTensors);
		outputTensorVector.close();
		outputTensorVector.deallocate();
		output.close();
		output.deallocate();
		for (int i = 0; i < inputsVector.size(); i ++) {
			inputsVector.get(i).close();
			inputsVector.get(i).deallocate();
		}
		inputsVector.close();
		inputsVector.deallocate();		
	}
	
	protected void runFromShmas(List<String> inputs, List<String> outputs) throws IOException {
		
		IValueVector inputsVector = new IValueVector();
		for (String ee : inputs) {
			Map<String, Object> decoded = Types.decode(ee);
			SharedMemoryArray shma = SharedMemoryArray.read((String) decoded.get(MEM_NAME_KEY));
			org.bytedeco.pytorch.Tensor  inT = TensorBuilder.build(shma);
        	inputsVector.put(new IValue(inT));
			if (PlatformDetection.isWindows()) shma.close();
		}
        // Run model
		model.eval();
        IValue output = model.forward(inputsVector);
        TensorVector outputTensorVector = null;
        if (output.isTensorList()) {
        	System.out.println("SSECRET_KEY :  1 ");
        	outputTensorVector = output.toTensorVector();
        } else {
        	System.out.println("SSECRET_KEY :  2 ");
        	outputTensorVector = new TensorVector();
        	outputTensorVector.put(output.toTensor());
        }

		// Fill the agnostic output tensors list with data from the inference result
		int c = 0;
		for (String ee : outputs) {
			Map<String, Object> decoded = Types.decode(ee);
			System.out.println("ENTERED: " + ee);
			ShmBuilder.build(outputTensorVector.get(c ++), (String) decoded.get(MEM_NAME_KEY));
		}
		outputTensorVector.close();
		outputTensorVector.deallocate();
		output.close();
		output.deallocate();
		for (int i = 0; i < inputsVector.size(); i ++) {
			inputsVector.get(i).close();
			inputsVector.get(i).deallocate();
		}
		inputsVector.close();
		inputsVector.deallocate();	
	}
	
	/**
	 * MEthod only used in MacOS Intel and Windows systems that makes all the arrangements
	 * to create another process, communicate the model info and tensors to the other 
	 * process and then retrieve the results of the other process
	 * @param inputTensors
	 * 	tensors that are going to be run on the model
	 * @param outputTensors
	 * 	expected results of the model
	 * @throws RunModelException if there is any issue running the model
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void runInterprocessing(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		shmaInputList = new ArrayList<SharedMemoryArray>();
		shmaOutputList = new ArrayList<SharedMemoryArray>();
		List<String> encIns = encodeInputs(inputTensors);
		List<String> encOuts = encodeOutputs(outputTensors);
		LinkedHashMap<String, Object> args = new LinkedHashMap<String, Object>();
		args.put("inputs", encIns);
		args.put("outputs", encOuts);

		try {
			Task task = runner.task("inference", args);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException();
			}
			for (int i = 0; i < outputTensors.size(); i ++) {
	        	String name = (String) Types.decode(encOuts.get(i)).get(MEM_NAME_KEY);
	        	SharedMemoryArray shm = shmaOutputList.stream()
	        			.filter(ss -> ss.getName().equals(name)).findFirst().orElse(null);
	        	if (shm == null) {
	        		shm = SharedMemoryArray.read(name);
	        		shmaOutputList.add(shm);
	        	}
	        	RandomAccessibleInterval<?> rai = shm.getSharedRAI();
	        	outputTensors.get(i).setData(Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(rai), Util.getTypeFromInterval(Cast.unchecked(rai))));
	        }
		} catch (Exception e) {
			closeShmas();
			if (e instanceof RunModelException)
				throw (RunModelException) e;
			throw new RunModelException(Types.stackTrace(e));
		}
		closeShmas();
	}
	
	private void closeShmas() {
		shmaInputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaInputList = null;
		shmaOutputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaOutputList = null;
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Closes the Pytorch model and sets it to null once the model is not needed anymore.
	 * 
	 */
	@Override
	public void closeModel() {
		if (this.interprocessing && runner != null) {
			Task task;
			try {
				task = runner.task("close");
				task.waitFor();
			} catch (IOException | InterruptedException e) {
				throw new RuntimeException(Types.stackTrace(e));
			}
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException();
			}
			this.runner.close();
			this.runner = null;
			return;
		} else if (this.interprocessing) {
			return;
		}
		if (model == null)
			return;
		model.close();
		model.deallocate();
		model = null;
	}

	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by the model-runner
	 * 
	 * @param tensorVector 
	 * 	an object containing a list of Pytorch JavaCpp tensors
	 * @param outputTensors 
	 * 	the list of output tensors where the output data is going to be written to send back
	 * 	to the model runner
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static <T extends RealType<T> & NativeType<T>> 
	void fillOutputTensors(TensorVector tensorVector, List<Tensor<T>> outputTensors) throws RunModelException{
		if (tensorVector.size() != outputTensors.size())
			throw new RunModelException((int) tensorVector.size(), outputTensors.size());
		for (int i = 0; i < tensorVector.size(); i ++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(tensorVector.get(i)));
			tensorVector.get(i).close();
			tensorVector.get(i).deallocate();
		}
	}
	
	
	private <T extends RealType<T> & NativeType<T>> List<String> encodeInputs(List<Tensor<T>> inputTensors) {
		List<String> encodedInputTensors = new ArrayList<String>();
		Gson gson = new Gson();
		for (Tensor<T> tt : inputTensors) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
			shmaInputList.add(shma);
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(SHAPE_KEY, tt.getShape());
			map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
			map.put(IS_INPUT_KEY, true);
			map.put(MEM_NAME_KEY, shma.getName());
			encodedInputTensors.add(gson.toJson(map));
		}
		return encodedInputTensors;
	}
	
	
	private <T extends RealType<T> & NativeType<T>> 
	List<String> encodeOutputs(List<Tensor<T>> outputTensors) {
		Gson gson = new Gson();
		List<String> encodedOutputTensors = new ArrayList<String>();
		for (Tensor<?> tt : outputTensors) {
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(IS_INPUT_KEY, false);
			if (!tt.isEmpty()) {
				map.put(SHAPE_KEY, tt.getShape());
				map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
				SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else if (PlatformDetection.isWindows()){
				SharedMemoryArray shma = SharedMemoryArray.create(0);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else {
				String memName = SharedMemoryArray.createShmName();
				map.put(MEM_NAME_KEY, memName);
				shmaNamesList.add(memName);
			}
			encodedOutputTensors.add(gson.toJson(map));
		}
		return encodedOutputTensors;
	}
	
	/**
	 * Create the arguments needed to execute tensorflow 2 in another 
	 * process with the corresponding tensors
	 * @return the command used to call the separate process
	 * @throws IOException if the command needed to execute interprocessing is too long
	 * @throws URISyntaxException if there is any error with the URIs retrieved from the classes
	 */
	private List<String> getProcessCommandsWithoutArgs() throws IOException, URISyntaxException {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +  File.separator + "bin" + File.separator + "java";

        String classpath = getCurrentClasspath();
        ProtectionDomain protectionDomain = PytorchJavaCPPInterface.class.getProtectionDomain();
        String codeSource = protectionDomain.getCodeSource().getLocation().getPath();
        String f_name = URLDecoder.decode(codeSource, StandardCharsets.UTF_8.toString());
        f_name = new File(f_name).getAbsolutePath();
        for (File ff : new File(f_name).getParentFile().listFiles()) {
        	if (ff.getName().startsWith(JAR_FILE_NAME) && !ff.getAbsolutePath().equals(f_name))
        		continue;
        	classpath += ff.getAbsolutePath() + File.pathSeparator;
        }
        String className = JavaWorker.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(padSpecialJavaBin(javaBin));
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        return command;
	}
	
    private static String getCurrentClasspath() throws UnsupportedEncodingException {

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        String gsonPath = getPathFromClass(Gson.class);
        String jnaPath = getPathFromClass(com.sun.jna.Library.class);
        String jnaPlatformPath = getPathFromClass(com.sun.jna.platform.FileUtils.class);
        if (modelrunnerPath == null || (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator)))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        classpath =  classpath + gsonPath + File.pathSeparator;
        classpath =  classpath + jnaPath + File.pathSeparator;
        classpath =  classpath + jnaPlatformPath + File.pathSeparator;

        return classpath;
    }
	
	/**
	 * Method that gets the path to the JAR from where a specific class is being loaded
	 * @param clazz
	 * 	class of interest
	 * @return the path to the JAR that contains the class
	 * @throws UnsupportedEncodingException if the url of the JAR is not encoded in UTF-8
	 */
	private static String getPathFromClass(Class<?> clazz) throws UnsupportedEncodingException {
	    String classResource = clazz.getName().replace('.', '/') + ".class";
	    URL resourceUrl = clazz.getClassLoader().getResource(classResource);
	    if (resourceUrl == null) {
	        return null;
	    }
	    String urlString = resourceUrl.toString();
	    if (urlString.startsWith("jar:")) {
	        urlString = urlString.substring(4);
	    }
	    if (urlString.startsWith("file:/") && PlatformDetection.isWindows()) {
	        urlString = urlString.substring(6);
	    } else if (urlString.startsWith("file:/") && !PlatformDetection.isWindows()) {
	        urlString = urlString.substring(5);
	    }
	    urlString = URLDecoder.decode(urlString, "UTF-8");
	    File file = new File(urlString);
	    String path = file.getAbsolutePath();
	    if (path.lastIndexOf(".jar!") != -1)
	    	path = path.substring(0, path.lastIndexOf(".jar!")) + ".jar";
	    return path;
	}
	
	/**
	 * if java bin dir contains any special char, surround it by double quotes
	 * @param javaBin
	 * 	java bin dir
	 * @return impored java bin dir if needed
	 */
	private static String padSpecialJavaBin(String javaBin) {
		String[] specialChars = new String[] {" "};
        for (String schar : specialChars) {
        	if (javaBin.contains(schar) && PlatformDetection.isWindows()) {
        		return "\"" + javaBin + "\"";
        	}
        }
        return javaBin;
	}
}
