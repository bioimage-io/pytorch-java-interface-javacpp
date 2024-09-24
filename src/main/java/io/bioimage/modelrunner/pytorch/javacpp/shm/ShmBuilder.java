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

import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import org.bytedeco.pytorch.Tensor;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link RandomAccessibleInterval} builder for TensorFlow {@link Tensor} objects.
 * Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
 * from Tensorflow 2 {@link Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class ShmBuilder
{
    /**
     * Utility class.
     */
    private ShmBuilder()
    {
    }

	/**
	 * Creates a {@link RandomAccessibleInterval} from a given {@link TType} tensor
	 * 
	 * @param <T> 
	 * 	the possible ImgLib2 datatypes of the image
	 * @param tensor 
	 * 	The {@link TType} tensor data is read from.
	 * @throws IllegalArgumentException If the {@link TType} tensor type is not supported.
	 * @throws IOException 
	 */
	public static void build(Tensor tensor, String memoryName) throws IllegalArgumentException, IOException
    {
        if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Byte)
    			|| tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Char)) {
        	buildFromTensorByte(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Int)) {
        	buildFromTensorInt(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Float)) {
        	buildFromTensorFloat(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Double)) {
        	buildFromTensorDouble(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Long)) {
        	buildFromTensorLong(tensor, memoryName);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.scalar_type());
    	}
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned byte-typed {@link TUint8} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TUint8} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link UnsignedByteType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorByte(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 1))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per ubyte output tensor supported: " + Integer.MAX_VALUE / 1);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	byte[] flatArr = new byte[(int) flatSize];
    	tensor.data_ptr_byte().get(flatArr);
		SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new UnsignedByteType(), false, true);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned int32-typed {@link TInt32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt32} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link IntType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorInt(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per int output tensor supported: " + Integer.MAX_VALUE / 4);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	int[] flatArr = new int[(int) flatSize];
    	tensor.data_ptr_int().get(flatArr);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer byteBuffer = ByteBuffer.allocate(flatArr.length * Integer.BYTES);
        byteBuffer.asIntBuffer().put(flatArr);
        shma.setBuffer(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned float32-typed {@link TFloat32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat32} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link FloatType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorFloat(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per float output tensor supported: " + Integer.MAX_VALUE / 4);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	float[] flatArr = new float[(int) flatSize];
    	// TODO check what we get with tensor.data_ptr_byte().get(flatArr);
    	tensor.data_ptr_float().get(flatArr);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer byteBuffer = ByteBuffer.allocate(flatArr.length * Float.BYTES);
        byteBuffer.asFloatBuffer().put(flatArr);
        shma.setBuffer(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned float64-typed {@link TFloat64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat64} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link DoubleType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorDouble(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per double output tensor supported: " + Integer.MAX_VALUE / 8);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	double[] flatArr = new double[(int) flatSize];
    	tensor.data_ptr_double().get(flatArr);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer byteBuffer = ByteBuffer.allocate(flatArr.length * Double.BYTES);
        byteBuffer.asDoubleBuffer().put(flatArr);
        shma.setBuffer(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned int64-typed {@link TInt64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt64} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link LongType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorLong(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per long output tensor supported: " + Integer.MAX_VALUE / 8);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	long[] flatArr = new long[(int) flatSize];
    	tensor.data_ptr_long().get(flatArr);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer byteBuffer = ByteBuffer.allocate(flatArr.length * Long.BYTES);
        byteBuffer.asLongBuffer().put(flatArr);
        shma.setBuffer(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }
}
