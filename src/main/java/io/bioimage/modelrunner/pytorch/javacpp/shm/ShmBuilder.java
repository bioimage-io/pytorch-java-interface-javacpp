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
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

import org.bytedeco.pytorch.Tensor;

import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A utility class that converts {@link Tensor}s into {@link SharedMemoryArray}s for
 * interprocessing communication
 * 
 * @author Carlos Garcia Lopez de Haro
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
     * Create a {@link SharedMemoryArray} from a {@link Tensor}
     * @param tensor
     * 	the tensor to be passed into the other process through the shared memory
     * @param memoryName
     * 	the name of the memory region where the tensor is going to be copied
     * @throws IllegalArgumentException if the data type of the tensor is not supported
     * @throws IOException if there is any error creating the shared memory array
     */
	public static void build(Tensor tensor, String memoryName) throws IllegalArgumentException, IOException
    {
        if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Byte)
    			|| tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Char)) {
        	System.out.println("SSECRET_KEY :  BYTE ");
        	buildFromTensorByte(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Int)) {
        	System.out.println("SSECRET_KEY :  INT ");
        	buildFromTensorInt(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Float)) {
        	System.out.println("SSECRET_KEY :  FLOAT ");
        	buildFromTensorFloat(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Double)) {
        	System.out.println("SSECRET_KEY :  SOUBKE ");
        	buildFromTensorDouble(tensor, memoryName);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Long)) {
        	System.out.println("SSECRET_KEY :  LONG ");
        	buildFromTensorLong(tensor, memoryName);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.scalar_type());
    	}
    }

    private static void buildFromTensorByte(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 1))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per ubyte output tensor supported: " + Integer.MAX_VALUE / 1);
		SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new ByteType(), false, true);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	byte[] flat = new byte[(int) flatSize];
    	ByteBuffer byteBuffer = ByteBuffer.allocateDirect((int) (flatSize));
    	tensor.data_ptr_byte().get(flat);
    	byteBuffer.put(flat);
    	byteBuffer.rewind();
        shma.getDataBufferNoHeader().put(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorInt(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per int output tensor supported: " + Integer.MAX_VALUE / 4);
		SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	int[] flat = new int[(int) flatSize];
    	ByteBuffer byteBuffer = ByteBuffer.allocateDirect((int) (flatSize * Integer.BYTES));
    	IntBuffer floatBuffer = byteBuffer.asIntBuffer();
    	tensor.data_ptr_int().get(flat);
    	floatBuffer.put(flat);
    	byteBuffer.rewind();
        shma.getDataBufferNoHeader().put(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorFloat(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per float output tensor supported: " + Integer.MAX_VALUE / 4);
		SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new FloatType(), false, true);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	float[] flat = new float[(int) flatSize];
    	ByteBuffer byteBuffer = ByteBuffer.allocateDirect((int) (flatSize * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
    	FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
    	tensor.data_ptr_float().get(flat);
    	floatBuffer.put(flat);
        shma.getDataBufferNoHeader().put(byteBuffer);
        System.out.println("equals  " + (shma.getDataBufferNoHeader().get(100) == byteBuffer.get(100)));
        System.out.println("equals  " + (shma.getDataBufferNoHeader().get(500) == byteBuffer.get(500)));
        System.out.println("equals  " + (shma.getDataBufferNoHeader().get(300) == byteBuffer.get(300)));
        System.out.println("equals  " + (shma.getDataBufferNoHeader().get(1000) == byteBuffer.get(1000)));
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorDouble(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per double output tensor supported: " + Integer.MAX_VALUE / 8);
		SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new DoubleType(), false, true);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	double[] flat = new double[(int) flatSize];
    	ByteBuffer byteBuffer = ByteBuffer.allocateDirect((int) (flatSize * Double.BYTES));
    	DoubleBuffer floatBuffer = byteBuffer.asDoubleBuffer();
    	tensor.data_ptr_double().get(flat);
    	floatBuffer.put(flat);
    	byteBuffer.rewind();
        shma.getDataBufferNoHeader().put(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorLong(Tensor tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per long output tensor supported: " + Integer.MAX_VALUE / 8);
		SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new LongType(), false, true);
		long flatSize = 1;
    	for (long l : arrayShape) {flatSize *= l;}
    	long[] flat = new long[(int) flatSize];
    	ByteBuffer byteBuffer = ByteBuffer.allocateDirect((int) (flatSize * Long.BYTES));
    	LongBuffer floatBuffer = byteBuffer.asLongBuffer();
    	tensor.data_ptr_long().get(flat);
    	floatBuffer.put(flat);
    	byteBuffer.rewind();
        shma.getDataBufferNoHeader().put(byteBuffer);
        if (PlatformDetection.isWindows()) shma.close();
    }
}
