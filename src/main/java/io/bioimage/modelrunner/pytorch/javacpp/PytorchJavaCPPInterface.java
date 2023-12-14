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

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Type;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import org.bytedeco.pytorch.IValue;
import org.bytedeco.pytorch.IValueVector;
import org.bytedeco.pytorch.JitModule;
import org.bytedeco.pytorch.TensorVector;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.pytorch.javacpp.shm.NDArrayShmBuilder;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.JavaCPPTensorBuilder;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

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
	
	private String modelSource;
	
	private boolean interprocessing = true;
	
	private Process process;
	
	private List<SharedMemoryArray> shmaList = new ArrayList<SharedMemoryArray>();
	
	private List<String> shmaNamesList = new ArrayList<String>();

	private static final String NAME_KEY = "name";
	private static final String SHAPE_KEY = "shape";
	private static final String DTYPE_KEY = "dtype";
	private static final String IS_INPUT_KEY = "isInput";
	private static final String MEM_NAME_KEY = "memoryName";

	/**
	 * Constructor for the interface. It is going to be called from the 
	 * dlmodel-runner.
	 * This one tries to use mkl to be faster
	 */
    public PytorchJavaCPPInterface()
    {
        //System.setProperty("org.bytedeco.openblas.load", "mkl");
    }
	
    /**
     * Private constructor that can only be launched from the class to create a separate
     * process to avoid the conflicts that occur in the same process between Pytorch1 and 2
     * @param doInterprocessing
     * 	whether to do interprocessing or not
     */
    private PytorchJavaCPPInterface(boolean doInterprocessing)
    {
    	interprocessing = doInterprocessing;
    }
    
    public static < T extends RealType< T > & NativeType< T > > void main(String[] args) throws LoadModelException, RunModelException {
    	if (args.length == 0) {
    		
	    	String modelFolder = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\Neuron Segmentation in EM (Membrane Prediction)_30102023_192607";
	    	String modelSourc = modelFolder + "\\weights-torchscript.pt";
	    	PytorchJavaCPPInterface pi = new PytorchJavaCPPInterface();
	    	pi.loadModel(modelFolder, modelSourc);
	    	RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(new long[] {1, 1, 16, 144, 144});
	    	Tensor<?> inp = Tensor.build("aa", "bczyx", rai);
	    	Tensor<?> out = Tensor.buildEmptyTensor("oo", "bczyx");
	    	List<Tensor<?>> ins = new ArrayList<Tensor<?>>();
	    	List<Tensor<?>> ous = new ArrayList<Tensor<?>>();
	    	ins.add(inp);
	    	ous.add(out);
	    	pi.run(ins, ous);
	    	System.out.println(false);
	    	System.gc();
	    	return;
    	}
    	// Unpack the args needed
    	 if (args.length < 3)
    		throw new IllegalArgumentException("Error exectuting Pytorch, "
    				+ "at least35 arguments are required:" + System.lineSeparator()
    				+ " - Path to the model weigths." + System.lineSeparator()
    				+ " - Encoded input 1" + System.lineSeparator()
    				+ " - Encoded input 2 (if exists)" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Encoded input n (if exists)" + System.lineSeparator()
    				+ " - Encoded output 1" + System.lineSeparator()
    				+ " - Encoded output 2 (if exists)" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Encoded output n (if exists)" + System.lineSeparator()
    				);
    	String modelSource = args[0];
    	if (!(new File(modelSource).isFile())) {
    		throw new IllegalArgumentException("Argument 0 of the main method, '" + modelSource + "' "
    				+ "should be the path to the wanted .pth weights file.");
    	}
    	PytorchJavaCPPInterface ptInterface = new PytorchJavaCPPInterface(false);
    	Gson gson = new Gson();
        Type mapType = new TypeToken<HashMap<String, Object>>() {}.getType();
    	try {
        	ptInterface.loadModel(new File(modelSource).getParent(), modelSource);
			IValueVector inputsVector = new IValueVector();
			for (int i = 1; i < args.length; i ++) {
	            HashMap<String, Object> map = gson.fromJson(args[i], mapType);
	            if ((boolean) map.get(IS_INPUT_KEY)) {
	            	RandomAccessibleInterval<T> rai = SharedMemoryArray.buildImgLib2FromNumpyLikeSHMA((String) map.get(MEM_NAME_KEY));
	            	inputsVector.put(new IValue(JavaCPPTensorBuilder.build(rai))); 
	            }
			}
	        // Run model
	        ptInterface.model.eval();
	        IValue output = ptInterface.model.forward(inputsVector);
	        TensorVector outputTensorVector = null;
	        if (output.isTensorList()) {
	        	outputTensorVector = output.toTensorVector();
	        } else {
	        	outputTensorVector = new TensorVector();
	        	outputTensorVector.put(output.toTensor());
	        }
			// Fill the agnostic output tensors list with data from the inference
			// result
			int c = 0;
			for (int i = 1; i < args.length; i ++) {
	            HashMap<String, Object> map = gson.fromJson(args[i], mapType);
				if (!((boolean) map.get(IS_INPUT_KEY))) {
					NDArrayShmBuilder.buildShma(outputTensorVector.get(c), (String) map.get(MEM_NAME_KEY));
					outputTensorVector.get(c).close();
					outputTensorVector.get(c).deallocate();
					c ++;
				}
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
			// Run model
		}
		catch (Exception e) {
			e.printStackTrace();
	    	ptInterface.closeModel();
			throw new RunModelException(e.toString());
		}
    	ptInterface.closeModel();
    }

	/**
	 * {@inheritDoc}
	 * 
	 * Run a Pytorch model using JavaCpp on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 * 
	 */
	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		if (interprocessing) {
			runInterprocessing(inputTensors, outputTensors);
			return;
		}
		IValueVector inputsVector = new IValueVector();
		List<String> inputListNames = new ArrayList<String>();
        for (Tensor<?> tt : inputTensors) {
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

	/**
	 * {@inheritDoc}
	 * 
     * Load a Pytorch torchscript model using JavaCpp. 
	 */
	@Override
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
		this.modelSource = modelSource;
		if (interprocessing) 
			return;

    	try {
	    	model = org.bytedeco.pytorch.global.torch.load(modelSource);
			model.eval();
    	} catch (Exception ex) {
    		throw new LoadModelException(ex.toString());
    	}
	}


	/**
	 * {@inheritDoc}
	 * 
	 * Closes the Pytorch model and sets it to null once the model is not needed anymore.
	 * 
	 */
	@Override
	public void closeModel() {
		if (model == null)
			return;
		model.close();
		model.deallocate();
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
	public static void fillOutputTensors(TensorVector tensorVector, List<Tensor<?>> outputTensors) throws RunModelException{
		if (tensorVector.size() != outputTensors.size())
			throw new RunModelException((int) tensorVector.size(), outputTensors.size());
		for (int i = 0; i < tensorVector.size(); i ++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(tensorVector.get(i)));
			tensorVector.get(i).close();
			tensorVector.get(i).deallocate();
		}
	}
	
	/**
	 * MEthod  that makes all the arrangements
	 * to create another process, communicate the model info and tensors to the other 
	 * process and then retrieve the results of the other process
	 * @param inputTensors
	 * 	tensors that are going to be run on the model
	 * @param outputTensors
	 * 	expected results of the model
	 * @throws RunModelException if there is any issue running the model
	 */
	public void runInterprocessing(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		shmaList = new ArrayList<SharedMemoryArray>();
		try {
			List<String> args = getProcessCommandsWithoutArgs();
			List<String> encIns = encodeInputs(inputTensors);
			args.addAll(modifyForWinCmd(encIns));
			List<String> encOuts = encodeOutputs(outputTensors);
			args.addAll(modifyForWinCmd(encOuts));
			//main(new String[] {modelSource, encIns.get(0), encOuts.get(0)});
			ProcessBuilder builder = new ProcessBuilder(args);
			builder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
			builder.redirectError(ProcessBuilder.Redirect.INHERIT);
	        process = builder.start();
	        int result = process.waitFor();
	        if (result != 0)
	    		throw new RunModelException("Error executing the Pytorch model in"
	        			+ " a separate process. The process was not terminated correctly."
	        			+ System.lineSeparator() + readProcessStringOutput(process));
	        process = null;
	        for (int i = 0; i < outputTensors.size(); i ++) {
	        	String name = (String) decodeString(encOuts.get(i)).get(MEM_NAME_KEY);
	        	outputTensors.get(i).setData(SharedMemoryArray.buildImgLib2FromNumpyLikeSHMA(name));
	        }
	        closeShmas();
		} catch (Exception e) {
			closeShmas();
			closeModel();
			throw new RunModelException(e.toString());
		}
	}
	
	private void closeShmas() {
		shmaList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		// TODO add methos imilar to Python's shared_memory.SharedMemory(name="") in SharedArrays class in JDLL
		this.shmaNamesList.forEach(shm -> {
			try { SharedMemoryArray.buildImgLib2FromNumpyLikeSHMA(shm); } catch (Exception e1) {}
		});
	}
	
	private static List<String> modifyForWinCmd(List<String> ins){
		if (!PlatformDetection.isWindows())
			return ins;
		List<String> newIns = new ArrayList<String>();
		for (String ii : ins)
			newIns.add("\"" + ii.replace("\"", "\\\"") + "\"");
		return newIns;
	}
	
	
	private List<String> encodeInputs(List<Tensor<?>> inputTensors) {
		int i = 0;
		List<String> encodedInputTensors = new ArrayList<String>();
		Gson gson = new Gson();
		for (Tensor<?> tt : inputTensors) {
			shmaList.add(SharedMemoryArray.buildNumpyLikeSHMA(tt.getData()));
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(SHAPE_KEY, tt.getShape());
			map.put(DTYPE_KEY, CommonUtils.getDataType(tt.getData()));
			map.put(IS_INPUT_KEY, true);
			map.put(MEM_NAME_KEY, shmaList.get(i).getName());
			encodedInputTensors.add(gson.toJson(map));
	        i ++;
		}
		return encodedInputTensors;
	}
	
	
	private List<String> encodeOutputs(List<Tensor<?>> outputTensors) {
		Gson gson = new Gson();
		List<String> encodedOutputTensors = new ArrayList<String>();
		for (Tensor<?> tt : outputTensors) {
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(IS_INPUT_KEY, false);
			if (!tt.isEmpty()) {
				map.put(SHAPE_KEY, tt.getShape());
				map.put(DTYPE_KEY, CommonUtils.getDataType(tt.getData()));
				SharedMemoryArray shma = SharedMemoryArray.buildNumpyLikeSHMA(tt.getData());
				shmaList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else if (PlatformDetection.isWindows()){
				String memName = SharedMemoryArray.createShmName();
				SharedMemoryArray shma = SharedMemoryArray.buildSHMA(memName, null);
				shmaList.add(shma);
				map.put(MEM_NAME_KEY, memName);
			} else {
				String memName = SharedMemoryArray.createShmName();
				map.put(MEM_NAME_KEY, memName);
				shmaNamesList.add(memName);
			}
			encodedOutputTensors.add(gson.toJson(map));
		}
		return encodedOutputTensors;
	}
	
	
	private HashMap<String, Object> decodeString(String encoded) {
		Gson gson = new Gson();
        Type mapType = new TypeToken<HashMap<String, Object>>() {}.getType();
        HashMap<String, Object> map = gson.fromJson(encoded, mapType);
		return map;
	}
	
	/**
	 * Create the arguments needed to execute Pytorch in another 
	 * process with the corresponding tensors
	 * @return the command used to call the separate process
	 * @throws IOException if the command needed to execute interprocessing is too long
	 * @throws URISyntaxException if there is any error with the URIs retrieved from the classes
	 */
	private List<String> getProcessCommandsWithoutArgs() throws IOException, URISyntaxException {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +  File.separator + "bin" + File.separator + "java";

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        if (modelrunnerPath == null || (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator)))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        ProtectionDomain protectionDomain = PytorchJavaCPPInterface.class.getProtectionDomain();
        String codeSource = protectionDomain.getCodeSource().getLocation().getPath();
        String f_name = URLDecoder.decode(codeSource, StandardCharsets.UTF_8.toString());
	        for (File ff : new File(f_name).getParentFile().listFiles()) {
	        	classpath += ff.getAbsolutePath() + File.pathSeparator;
	        }
        String className = PytorchJavaCPPInterface.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(padSpecialJavaBin(javaBin));
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        command.add(modelSource);
        return command;
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
    
    /**
     * MEthod to obtain the String output of the process in case something goes wrong
     * @param process
     * 	the process that executed the TF2 model
     * @return the String output that we would have seen on the terminal
     * @throws IOException if the output of the terminal cannot be seen
     */
    private static String readProcessStringOutput(Process process) throws IOException {
    	BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
		BufferedReader bufferedErrReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
		String text = "";
		String line;
	    while ((line = bufferedErrReader.readLine()) != null) {
	    	text += line + System.lineSeparator();
	    }
	    while ((line = bufferedReader.readLine()) != null) {
	    	text += line + System.lineSeparator();
	    }
	    return text;
    }
}
