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


import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
* A {@link RandomAccessibleInterval} builder for JAvaCPP Pytorch {@link org.bytedeco.pytorch.Tensor} objects.
* Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
* from Pytorch {@link org.bytedeco.pytorch.Tensor}
* 
* @author Carlos Garcia Lopez de Haro
*/
public class ImgLib2Builder {

	/**
	 * Creates a {@link RandomAccessibleInterval} from a given {@link org.bytedeco.pytorch.Tensor} 
	 *  
	 * @param <T>
	 * 	the ImgLib2 data type that the {@link RandomAccessibleInterval} can have
	 * @param tensor
	 * 	the {@link org.bytedeco.pytorch.Tensor} that wants to be converted
	 * @return the {@link RandomAccessibleInterval} that resulted from the {@link org.bytedeco.pytorch.Tensor} 
	 * @throws IllegalArgumentException if the dataype of the {@link org.bytedeco.pytorch.Tensor} 
	 * is not supported
	 */
    @SuppressWarnings("unchecked")
	public static <T extends Type<T>> RandomAccessibleInterval<T> build(org.bytedeco.pytorch.Tensor tensor) throws IllegalArgumentException
    {
        if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Byte)
    			|| tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Char)) {
    		return (RandomAccessibleInterval<T>) buildFromTensorByte(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Int)) {
    		return (RandomAccessibleInterval<T>) buildFromTensorInt(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Float)) {
    		return (RandomAccessibleInterval<T>) buildFromTensorFloat(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Double)) {
    		return (RandomAccessibleInterval<T>) buildFromTensorDouble(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Long)) {
            return (RandomAccessibleInterval<T>) buildFromTensorLong(tensor);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.scalar_type());
    	}
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned byte-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link UnsignedByteType}.
	 */
    public static RandomAccessibleInterval<UnsignedByteType> buildFromTensorUByte(org.bytedeco.pytorch.Tensor tensor) {
    	long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	byte[] flatArr = new byte[(int) flatSize];
    	tensor.data_ptr_byte().get(flatArr);
		RandomAccessibleInterval<UnsignedByteType> rai = ArrayImgs.unsignedBytes(flatArr, tensorShape);
		return Utils.transpose(rai);
	}


	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed byte-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link ByteType}.
	 */
    public static RandomAccessibleInterval<ByteType> buildFromTensorByte(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	byte[] flatArr = new byte[(int) flatSize];
    	tensor.data_ptr_byte().get(flatArr);
		RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(flatArr, tensorShape);
		return Utils.transpose(rai);
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed integer-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link IntType}.
	 */
    public static RandomAccessibleInterval<IntType> buildFromTensorInt(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	int[] flatArr = new int[(int) flatSize];
    	tensor.data_ptr_int().get(flatArr);
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		return Utils.transpose(rai);
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed float-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link FloatType}.
	 */
    public static RandomAccessibleInterval<FloatType> buildFromTensorFloat(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	float[] flatArr = new float[(int) flatSize];
    	tensor.data_ptr_float().get(flatArr);
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		return Utils.transpose(rai);
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed double-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link DoubleType}.
	 */
    public static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	double[] flatArr = new double[(int) flatSize];
    	tensor.data_ptr_double().get(flatArr);
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return Utils.transpose(rai);
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed long-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link LongType}.
	 */
    public static RandomAccessibleInterval<LongType> buildFromTensorLong(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	long[] flatArr = new long[(int) flatSize];
    	tensor.data_ptr_long().get(flatArr);
		RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(flatArr, tensorShape);
		return Utils.transpose(rai);
    }
}
