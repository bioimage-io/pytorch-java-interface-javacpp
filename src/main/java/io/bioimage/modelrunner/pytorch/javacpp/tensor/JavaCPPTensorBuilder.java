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
import io.bioimage.modelrunner.tensor.Utils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.blocks.PrimitiveBlocks;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

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
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< ByteType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
	 	org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, ogShape);
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
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< IntType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, ogShape);
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
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< FloatType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, ogShape);
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
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< DoubleType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		org.bytedeco.pytorch.Tensor ndarray = org.bytedeco.pytorch.Tensor.create(flatArr, ogShape);
	 	return ndarray;
    }
}
