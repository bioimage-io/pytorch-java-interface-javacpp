package io.bioimage.modelrunner.pytorch.javacpp;

import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.pytorch.IValue;
import org.bytedeco.pytorch.IValueVector;
import org.bytedeco.pytorch.JitModule;
import org.bytedeco.pytorch.TensorVector;

import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.pytorch.javacpp.tensor.JavaCPPTensorBuilder;
import io.bioimage.modelrunner.tensor.Tensor;


/**
 * This class implements an interface that allows the main plugin
 * to interact in an agnostic way with the Deep Learning model and
 * tensors to make inference.
 * This implementation add the Pytorch support to the main program.
 * 
 * @see SequenceBuilder SequenceBuilder: Create sequences from tensors.
 * @see JavaCPPTensorBuilder TensorBuilder: Create tensors from images and sequences.
 * @author Carlos Garcia Lopez de Haro 
 */
public class PytorchJavaCPPInterface implements DeepLearningEngineInterface
{
	private JitModule model;
    
    public PytorchJavaCPPInterface()
    {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
    }
    
    public static void main(String[] args) {
    }

	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		IValueVector inputsVector = new IValueVector();
		List<String> inputListNames = new ArrayList<String>();
        for (Tensor<?> tt : inputTensors) {
        	inputListNames.add(tt.getName());
        	inputsVector.put(new IValue(JavaCPPTensorBuilder.build(tt)));
        }
        // Run model
        IValue output = model.forward(inputsVector);
        TensorVector outputTensorVector = output.toTensorVector();
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

	@Override
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
    	try {
	    	model = org.bytedeco.pytorch.global.torch.load(modelSource);
			model.eval();
    	} catch (Exception ex) {
    		throw new LoadModelException(ex.toString());
    	}
	}

	@Override
	public void closeModel() {
		if (model == null)
			return;
		model.close();
		model.deallocate();
	}
	
	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning engine
	 * that can be readable by Deep Icy
	 * @param outputTensors
	 * 	an NDList containing NDArrays (tensors)
	 * @param outputTensors2
	 * 	the names given to the tensors by the model
	 * @return a list with Deep Learning framework agnostic tensors
	 * @throws RunModelException If the number of tensors expected is not the same as the number of
	 * 	Tensors outputed by the model
	 */
	public static void fillOutputTensors(TensorVector outputNDArrays, List<Tensor<?>> outputTensors) throws RunModelException{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException((int) outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i ++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
			outputNDArrays.get(i).close();
			outputNDArrays.get(i).deallocate();
		}
	}
	
	/**
	 * Print the correct message depending on the exception produced when
	 * trying to load the model
	 * 
	 * @param ex
	 * 	the exception that occurred
	 */
	public static void managePytorchExceptions(Exception e) {
		if (e instanceof MalformedURLException) {
			System.out.println("No model was found in the folder provided.");
		} else if (e instanceof Exception) {
			String err = e.getMessage();
			String os = System.getProperty("os.name").toLowerCase();
			String msg;
			if (os.contains("win") && err.contains("https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md")) {
				msg = "DeepIcy could not load the model.\n" + 
					"Please install the Visual Studio 2019 redistributables and reboot" +
					"your machine to be able to use Pytorch with DeepIcy.\n" +
					"For more information:\n" +
					" -https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md\n" +
					" -https://github.com/awslabs/djl/issues/126\n" +
					"If you already have installed VS2019 redistributables, the error" +
					"might be caused by a missing dependency or an incompatible Pytorch version.\n" + 
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto).\n" +
					"should be compatible with each other." +
					"Please check the DeepIcy Wiki.";
			} else if((os.contains("linux") || os.contains("unix")) && err.contains("https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md")){
				msg  = "DeepIcy could not load the model.\n" +
					"Check that there are no repeated dependencies on the jars folder.\n" +
					"The problem might be caused by a missing or repeated dependency or an incompatible Pytorch version.\n" +
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " +
					"should be compatible with each other.\n" +
					"If the problem persists, please check the DeepIcy Wiki.";
			} else {
				msg  = "DeepIcy could not load the model.\n" +
					"Either the DJL Pytorch version is incompatible with the Torchscript model's " +
					"Pytorch version or the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " + 
					"are not compatible with each other.\n" +
					"Please check the DeepIcy Wiki.";
			}
			System.out.println(msg);
		} else if (e instanceof IOException) {
			String msg = "DeepImageJ could not load the model.\n" + 
				"The model provided is not a correct Torchscript model.";
			System.out.println(msg);
		} else if (e instanceof IOException) {
			System.out.println("An error occurred accessing the model file.");
		}
	}
	
	public void finalize() {
		System.out.println("Collected Garbage");
	}
}
