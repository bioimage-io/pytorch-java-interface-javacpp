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
package io.bioimage.modelrunner.pytorch.javacpp.tensor;

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

/**
 * Class that manages the creation of JAvaCPP Pytorch tensors  
 * {@link org.bytedeco.pytorch.Tensor} from JDLL tensors {@link Tensor} that
 * use ImgLib2 {@link RandomAccessibleInterval} as the backend
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class JavaCPPTensorBuilder {

	/**
	 * Creates a {@link org.bytedeco.pytorch.Tensor} from a given {@link Tensor}. 
	 * The {@link Tensor} contains the data and info(dimensions, dataype) 
	 * necessary to build a {@link org.bytedeco.pytorch.Tensor}
	 * @param tensor
	 * 	The {@link Tensor} that will be copied into a {@link org.bytedeco.pytorch.Tensor}
	 * @return The {@link org.bytedeco.pytorch.Tensor} built from the {@link Tensor}.
	 * @throws IllegalArgumentException if the tensor type is not supported
	 */
    public static org.bytedeco.pytorch.Tensor build(Tensor tensor) throws IllegalArgumentException
    {
    	if (Util.getTypeFromInterval(tensor.getData()) instanceof ByteType) {
            return buildFromTensorByte( tensor.getData());
    	} else if (Util.getTypeFromInterval(tensor.getData()) instanceof IntType) {
            return buildFromTensorInt( tensor.getData());
    	} else if (Util.getTypeFromInterval(tensor.getData()) instanceof FloatType) {
            return buildFromTensorFloat( tensor.getData());
    	} else if (Util.getTypeFromInterval(tensor.getData()) instanceof DoubleType) {
            return buildFromTensorDouble( tensor.getData());
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDataType());
    	}
    }

	/**
	 * Creates a {@link org.bytedeco.pytorch.Tensor} from a given {@link RandomAccessibleInterval}.
	 * 
	 * @param <T>
	 * 	possible ImgLib2 datatypes of the {@link RandomAccessibleInterval}
	 * @param tensor
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link org.bytedeco.pytorch.Tensor}
	 * @return The {@link org.bytedeco.pytorch.Tensor} built from the {@link RandomAccessibleInterval}.
	 * @throws IllegalArgumentException if the {@link RandomAccessibleInterval} is not supported
	 */
    public static <T extends Type<T>> org.bytedeco.pytorch.Tensor build(RandomAccessibleInterval<T> tensor) throws IllegalArgumentException
    {
    	if (Util.getTypeFromInterval(tensor) instanceof ByteType) {
            return buildFromTensorByte( (RandomAccessibleInterval<ByteType>) tensor);
    	} else if (Util.getTypeFromInterval(tensor) instanceof IntType) {
            return buildFromTensorInt( (RandomAccessibleInterval<IntType>) tensor);
    	} else if (Util.getTypeFromInterval(tensor) instanceof ByteType) {
            return buildFromTensorFloat( (RandomAccessibleInterval<FloatType>) tensor);
    	} else if (Util.getTypeFromInterval(tensor) instanceof DoubleType) {
            return buildFromTensorDouble( (RandomAccessibleInterval<DoubleType>) tensor);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + Util.getTypeFromInterval(tensor).getClass().toString());
    	}
    }

	/**
	 * Builds a {@link org.bytedeco.pytorch.Tensor} from a signed byte-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link org.bytedeco.pytorch.Tensor}
	 * @return The {@link org.bytedeco.pytorch.Tensor} built from the tensor of type {@link ByteType}.
	 */
    private static org.bytedeco.pytorch.Tensor buildFromTensorByte(RandomAccessibleInterval<ByteType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<ByteType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<ByteType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<ByteType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		byte[] flatArr = new byte[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = tensorCursor.get().getByte();
        	flatArr[flatPos] = val;
		}
	 	org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, tensorShape);
	 	return ndarray;
	}

	/**
	 * Builds a {@link org.bytedeco.pytorch.Tensor} from a signed integer-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link org.bytedeco.pytorch.Tensor}
	 * @return The {@link org.bytedeco.pytorch.Tensor} built from the tensor of type {@link IntType}.
	 */
    private static org.bytedeco.pytorch.Tensor buildFromTensorInt(RandomAccessibleInterval<IntType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<IntType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<IntType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<IntType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		int[] flatArr = new int[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = tensorCursor.get().getInteger();
        	flatArr[flatPos] = val;
		}
		org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, tensorShape);
	 	return ndarray;
    }

	/**
	 * Builds a {@link org.bytedeco.pytorch.Tensor} from a signed float-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link org.bytedeco.pytorch.Tensor}
	 * @return The {@link org.bytedeco.pytorch.Tensor} built from the tensor of type {@link FloatType}.
	 */
    private static org.bytedeco.pytorch.Tensor buildFromTensorFloat(RandomAccessibleInterval<FloatType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<FloatType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<FloatType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<FloatType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		float[] flatArr = new float[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = tensorCursor.get().getRealFloat();
        	flatArr[flatPos] = val;
		}
		org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, tensorShape);
	 	return ndarray;
    }

	/**
	 * Builds a {@link org.bytedeco.pytorch.Tensor} from a signed double-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link NDArray}
	 * @return The {@link org.bytedeco.pytorch.Tensor} built from the tensor of type {@link DoubleType}.
	 */
    private static org.bytedeco.pytorch.Tensor buildFromTensorDouble(RandomAccessibleInterval<DoubleType> tensor)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
    	Cursor<DoubleType> tensorCursor;
		if (tensor instanceof IntervalView)
			tensorCursor = ((IntervalView<DoubleType>) tensor).cursor();
		else if (tensor instanceof Img)
			tensorCursor = ((Img<DoubleType>) tensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : tensor.dimensionsAsLongArray()) { flatSize *= dd;}
		double[] flatArr = new double[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = tensorCursor.get().getRealDouble();
        	flatArr[flatPos] = val;
		}
		org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, tensorShape);
	 	return ndarray;
    }
}
