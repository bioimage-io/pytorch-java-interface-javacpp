/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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

package io.bioimage.modelrunner.pytorch.javacpp.shm;

import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import org.bytedeco.pytorch.Tensor;

/**
 * Utility class to build Pytorch Bytedeco tensors from shm segments using {@link SharedMemoryArray}
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class TensorBuilder {

	/**
	 * Utility class.
	 */
	private TensorBuilder() {}

	/**
	 * Creates {@link Tensor} instance from a {@link SharedMemoryArray}
	 * 
	 * @param array
	 * 	the {@link SharedMemoryArray} that is going to be converted into
	 *  a {@link Tensor} tensor
	 * @return the Pytorch {@link Tensor} as the one stored in the shared memory segment
	 * @throws IllegalArgumentException if the type of the {@link SharedMemoryArray}
	 *  is not supported
	 */
	public static Tensor build(SharedMemoryArray array) throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (array.getOriginalDataType().equals("int8")) {
			return buildByte(array);
		}
		else if (array.getOriginalDataType().equals("int32")) {
			return buildInt(array);
		}
		else if (array.getOriginalDataType().equals("float32")) {
			return buildFloat(array);
		}
		else if (array.getOriginalDataType().equals("float64")) {
			return buildDouble(array);
		}
		else if (array.getOriginalDataType().equals("int64")) {
			return buildLong(array);
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + array.getOriginalDataType());
		}
	}

	private static Tensor buildByte(SharedMemoryArray shmArray)
		throws IllegalArgumentException
	{
		long[] ogShape = shmArray.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!shmArray.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = shmArray.getDataBufferNoHeader();
		byte[] flat = new byte[buff.capacity()];
		buff.get(flat);
		buff.rewind();
		Tensor ndarray = Tensor.create(flat, ogShape);
		return ndarray;
	}

	private static Tensor buildInt(SharedMemoryArray shmaArray)
		throws IllegalArgumentException
	{
		long[] ogShape = shmaArray.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!shmaArray.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = shmaArray.getDataBufferNoHeader();
		int[] flat = new int[buff.capacity() / 4];
		buff.asIntBuffer().get(flat);
		buff.rewind();
		Tensor ndarray = Tensor.create(flat, ogShape);
		return ndarray;
	}

	private static org.bytedeco.pytorch.Tensor buildLong(SharedMemoryArray shmArray)
		throws IllegalArgumentException
	{
		long[] ogShape = shmArray.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!shmArray.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = shmArray.getDataBufferNoHeader();
		long[] flat = new long[buff.capacity() / 8];
		buff.asLongBuffer().get(flat);
		buff.rewind();
		Tensor ndarray = Tensor.create(flat, ogShape);
		return ndarray;
	}

	private static org.bytedeco.pytorch.Tensor buildFloat(SharedMemoryArray shmArray)
		throws IllegalArgumentException
	{
		long[] ogShape = shmArray.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!shmArray.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = shmArray.getDataBufferNoHeader().order(ByteOrder.BIG_ENDIAN);
		float[] flat = new float[buff.capacity() / 4];
		buff.asFloatBuffer().get(flat);
		Tensor ndarray = Tensor.create(flat, ogShape);

    	// TODO remove
    	double max0 = 0;
    	for (float ff : flat) {
    		if (ff > max0)
    			max0 = ff;
    	}
    	System.out.println("Input max: " + max0);
    	// TODO remove
		return ndarray;
	}

	private static org.bytedeco.pytorch.Tensor buildDouble(SharedMemoryArray shmArray)
		throws IllegalArgumentException
	{
		long[] ogShape = shmArray.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!shmArray.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = shmArray.getDataBufferNoHeader();
		double[] flat = new double[buff.capacity() / 8];
		buff.asDoubleBuffer().get(flat);
		buff.rewind();
		Tensor ndarray = Tensor.create(flat, ogShape);
		return ndarray;
	}
}
