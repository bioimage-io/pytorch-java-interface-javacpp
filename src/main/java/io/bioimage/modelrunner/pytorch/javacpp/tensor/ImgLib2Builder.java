/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Pytorch using JavaCPP.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.pytorch.javacpp.tensor;

import org.bytedeco.pytorch.Tensor;

import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
* A {@link Img} builder for JAvaCPP Pytorch {@link org.bytedeco.pytorch.Tensor} objects.
* Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
* from Pytorch {@link org.bytedeco.pytorch.Tensor}
* 
* @author Carlos Garcia Lopez de Haro
*/
public class ImgLib2Builder {

	/**
	 * Creates a {@link Img} from a given {@link org.bytedeco.pytorch.Tensor} 
	 *  
	 * @param <T>
	 * 	the ImgLib2 data type that the {@link Img} can have
	 * @param tensor
	 * 	the {@link org.bytedeco.pytorch.Tensor} that wants to be converted
	 * @return the {@link Img} that resulted from the {@link org.bytedeco.pytorch.Tensor} 
	 * @throws IllegalArgumentException if the dataype of the {@link org.bytedeco.pytorch.Tensor} 
	 * is not supported
	 */
    public static <T extends Type<T>> Img<T> build(org.bytedeco.pytorch.Tensor tensor) throws IllegalArgumentException
    {
        if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Byte)
    			|| tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Char)) {
    		return (Img<T>) buildFromTensorByte(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Int)) {
    		return (Img<T>) buildFromTensorInt(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Float)) {
    		return (Img<T>) buildFromTensorFloat(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Double)) {
    		return (Img<T>) buildFromTensorDouble(tensor);
    	} else if (tensor.dtype().isScalarType(org.bytedeco.pytorch.global.torch.ScalarType.Long)) {
            return (Img<T>) buildFromTensorLong(tensor);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.scalar_type());
    	}
    }


	/**
	 * Builds a {@link Img} from a signed byte-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] tensorShape = tensor.shape();
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
    	long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	byte[] flatArr = new byte[(int) flatSize];
    	tensor.data_ptr_byte().get(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

	/**
	 * Builds a {@link Img} from a signed integer-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] tensorShape = tensor.shape();
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
    	long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	int[] flatArr = new int[(int) flatSize];
    	tensor.data_ptr_int().get(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

	/**
	 * Builds a {@link Img} from a signed float-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] tensorShape = tensor.shape();
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
    	long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	float[] flatArr = new float[(int) flatSize];
    	tensor.data_ptr_float().get(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

	/**
	 * Builds a {@link Img} from a signed double-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] tensorShape = tensor.shape();
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
    	long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	double[] flatArr = new double[(int) flatSize];
    	tensor.data_ptr_double().get(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

	/**
	 * Builds a {@link Img} from a signed long-typed {@link org.bytedeco.pytorch.Tensor}.
	 * 
	 * @param tensor 
	 * 	The {@link org.bytedeco.pytorch.Tensor} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(org.bytedeco.pytorch.Tensor tensor)
    {
    	long[] tensorShape = tensor.shape();
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
    	long flatSize = 1;
    	for (long l : tensorShape) {flatSize *= l;}
    	long[] flatArr = new long[(int) flatSize];
    	tensor.data_ptr_long().get(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	long val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }
}
