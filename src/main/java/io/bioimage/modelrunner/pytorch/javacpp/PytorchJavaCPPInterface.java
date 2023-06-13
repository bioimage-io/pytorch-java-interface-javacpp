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
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;

import org.bytedeco.pytorch.IValue;
import org.bytedeco.pytorch.IValueVector;
import org.bytedeco.pytorch.JitModule;
import org.bytedeco.pytorch.TensorVector;

import com.google.gson.Gson;

import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.JavaCPPTensorBuilder;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.mappedbuffer.ImgLib2ToMappedBuffer;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.mappedbuffer.MappedBufferToImgLib2;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;

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
     * Idetifier for the files that contain the data of the inputs
     */
    final private static String INPUT_PREFIX = "model_input";
    /**
     * Idetifier for the files that contain the data of the outputs
     */
    final private static String OUTPUT_PREFIX = "model_output";
    /**
     * Idetifier for the files that contain the data of the outputs
     */
    final private static String ERROR_PREFIX = "INTERPROCESSING_ERROR";
    /*
     * String that separates the prefic from the name from the data
     */
    final private static String SEPARATOR = "::--??";
	/**
	 * The Pytorch torchscript model loaded with JavaCpp
	 */
	private JitModule model;
	/**
	 * Whether the execution needs interprocessing (MacOS Interl) or not
	 */
	private boolean interprocessing = false;
    /**
     * Source file that contains the torchscript model
     */
    private String modelSource;
    /**
     * Write to the stdout to send data to the other process
     */
	private PrintWriter stdin;

    
    /**
     * TODO the interprocessing is executed for every OS
     * Constructor that detects whether the operating system where it is being 
     * executed is Windows or Mac or not to know if it is going to need interprocessing 
     * or not
     * @throws IOException if the temporary dir is not found
     */
    public PytorchJavaCPPInterface() throws IOException
    {
    	boolean isWin = PlatformDetection.isWindows();
    	boolean isIntel = new PlatformDetection().getArch().equals(PlatformDetection.ARCH_X86_64);
    	if (true || (isWin && isIntel)) {
    		interprocessing = true;
    	}
    }
	
    /**
     * Private constructor that can only be launched from the class to create a separate
     * process to avoid the conflicts that occur in the same process between TF2 and TF1/Pytorch
     * in Windows and Mac
     * @param doInterprocessing
     * 	whether to do interprocessing or not
     * @throws IOException if the temp dir is not found
     */
    private PytorchJavaCPPInterface(boolean doInterprocessing) throws IOException
    {
    	if (!doInterprocessing) {
    		interprocessing = false;
    	} else {
            System.setProperty("org.bytedeco.openblas.load", "mkl");
    		boolean isMac = PlatformDetection.isMacOS();
        	boolean isIntel = new PlatformDetection().getArch().equals(PlatformDetection.ARCH_X86_64);
        	if (isMac && isIntel) {
        		interprocessing = true;     		
        	}
    	}
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
	 * MEthod only used in MacOS Intel and Windows systems that makes all the arrangements
	 * to create another process, communicate the model info and tensors to the other 
	 * process and then retrieve the results of the other process
	 * @param inputTensors
	 * 	tensors that are going to be run on the model
	 * @param outputTensors
	 * 	expected results of the model
	 * @throws RunModelException if there is any issue running the model
	 */
	public void runInterprocessing(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		try {
			List<String> args = getProcessCommandsWithoutArgs();
			for (Tensor tensor : inputTensors) {
				args.add(INPUT_PREFIX + SEPARATOR + tensor2str(tensor));
				}
			for (Tensor tensor : outputTensors) {
				args.add(OUTPUT_PREFIX + SEPARATOR + tensor2str(tensor));
				}
			ProcessBuilder builder = new ProcessBuilder(args);
	        Process process = builder.start();
			BufferedReader stdout = new BufferedReader(new InputStreamReader(process.getInputStream()));
	        while (true) {
	        	break;
	        }
			if (process.waitFor() != 0)
	    		throw new RunModelException("Error executing the Tensorflow 2 model in"
	        			+ " a separate process. The process was not terminated correctly."
	        			+ System.lineSeparator() + readProcessStringOutput(process));
		} catch (RunModelException e) {
			closeModel();
			throw e;
		} catch (Exception e) {
			closeModel();
			throw new RunModelException(e.toString());
		}
		
		retrieveInterprocessingTensors(outputTensors);
	}
	
    /**
     * Create a temporary file for each of the tensors in the list to communicate with 
     * the separate process in MacOS Intel and Windows systems
     * @param tensors
     * 	list of tensors to be sent
     * @throws RunModelException if there is any error converting the tensors
     */
	private String tensor2str(Tensor<?> tensor) throws RunModelException{
		long lenFile = ImgLib2ToMappedBuffer.findTotalLengthFile(tensor);
		ByteBuffer byteBuffer = ByteBuffer.allocate((int) lenFile);
		ImgLib2ToMappedBuffer.build(tensor, byteBuffer);
		return new String(byteBuffer.array(), Charset.forName("ISO-8859-1"));
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

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        if (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        ProtectionDomain protectionDomain = Tensorflow2Interface.class.getProtectionDomain();
        CodeSource codeSource = protectionDomain.getCodeSource();
        for (File ff : new File(codeSource.getLocation().getPath()).getParentFile().listFiles()) {
        	classpath += ff.getAbsolutePath() + File.pathSeparator;
        }
        String className = Tensorflow2Interface.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(javaBin);
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        command.add(modelSource);
        return command;
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
	
	/**
	 * Retrieves the data of the tensors contained in the input list from the output
	 * generated by the independent process
	 * @param tensors
	 * 	list of tensors that are going to be filled
	 * @throws RunModelException if there is any issue retrieving the data from the other process
	 */
	private void retrieveInterprocessingTensors(List<Tensor<?>> tensors) throws RunModelException{
		for (Tensor<?> tensor : tensors) {
			ByteBuffer byteBuffer = null;
			tensor.setData(MappedBufferToImgLib2.build(byteBuffer));
		}
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
	 * Methods to run interprocessing and bypass the errors that occur in MacOS intel
	 * with the compatibility between TF2 and TF1/Pytorch
	 * This method checks that the arguments are correct, retrieves the input and output
	 * tensors, loads the model, makes inference with it and finally sends the tensors
	 * to the original process
     * 
     * @param args
     * 	arguments of the program:
     * 		- Path to the model folder
     * 		- Path to a temporary dir
     * 		- Name of the input 0
     * 		- Name of the input 1
     * 		- ...
     * 		- Name of the output n
     * 		- Name of the output 0
     * 		- Name of the output 1
     * 		- ...
     * 		- Name of the output n
     * @throws LoadModelException if there is any error loading the model
     * @throws IOException	if there is any error reading or writing any file or with the paths
     * @throws RunModelException	if there is any error running the model
     */
    public static void main(String[] args) throws LoadModelException, IOException, RunModelException {
    	// Unpack the args needed   	
    	if (args.length < 3)
    		throw new IllegalArgumentException("Error exectuting Pytorch, "
    				+ "at least 3 arguments are required:" + System.lineSeparator()
    				+ " - File path to the torchscript model" + System.lineSeparator()
    				+ " - Name of the model input followed by the String + '_model_input'" + System.lineSeparator()
    				+ " - Name of the second model input (if it exists) followed by the String + '_model_input'" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Name of the nth model input (if it exists)  followed by the String + '_model_input'" + System.lineSeparator()
    				+ " - Name of the model output followed by the String + '_model_output'" + System.lineSeparator()
    				+ " - Name of the second model output (if it exists) followed by the String + '_model_output'" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Name of the nth model output (if it exists)  followed by the String + '_model_output'" + System.lineSeparator()
    				);
    	String modelFolder = args[0];
    	if (!(new File(modelFolder).isDirectory())) {
    		throw new IllegalArgumentException("Argument 0 of the main method, '" + modelFolder + "' "
    				+ "should be an existing directory containing a Tensorflow 2 model.");
    	}
    	
    	PytorchJavaCPPInterface tfInterface = new PytorchJavaCPPInterface(false);
    	
    	tfInterface.stdin = new PrintWriter(System.out);
    	
    	tfInterface.loadModel(modelFolder, modelFolder);

    	List<Tensor<?>> inputList = new ArrayList<Tensor<?>>();
    	List<Tensor<?>> outputList = new ArrayList<Tensor<?>>();
    	for (int i = 1; i < args.length; i ++) {
    		if (args[i].startsWith(INPUT_PREFIX + SEPARATOR)){
    			byte[] arr = args[i].substring((INPUT_PREFIX + SEPARATOR).length(),
    							args[i].length()).getBytes(Charset.forName("ISO-8859-1"));
    			inputList.add(MappedBufferToImgLib2.buildTensor(ByteBuffer.wrap(arr)));
    		} else if (args[i].startsWith(OUTPUT_PREFIX + SEPARATOR)) {
    			byte[] arr = args[i].substring((OUTPUT_PREFIX + SEPARATOR).length(),
						args[i].length()).getBytes(Charset.forName("ISO-8859-1"));
    			outputList.add(MappedBufferToImgLib2.buildTensor(ByteBuffer.wrap(arr)));
    		}
    	}
    	tfInterface.run(inputList, outputList);
    	LinkedHashMap<String, Object> outMap = new LinkedHashMap<String, Object>();
    	int cc = 0;
    	for (Tensor<?> tt : outputList) {
    		long lenFile = ImgLib2ToMappedBuffer.findTotalLengthFile(tt);
    		ByteBuffer byteBuffer = ByteBuffer.allocate((int) lenFile);
    		ImgLib2ToMappedBuffer.build(tt, byteBuffer);
    		outMap.put(OUTPUT_PREFIX + cc, byteBuffer.array());
    	}
    	Gson gson = new Gson();
        String json = gson.toJson(outMap);
        tfInterface.stdin.println(json);
        tfInterface.stdin.flush();
    }
}
