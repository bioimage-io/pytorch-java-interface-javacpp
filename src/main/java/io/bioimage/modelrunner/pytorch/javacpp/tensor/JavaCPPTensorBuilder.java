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


public class JavaCPPTensorBuilder {


    /**
     * Creates a {@link NDArray} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    public static org.bytedeco.pytorch.Tensor build(Tensor tensor) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
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
     * Creates a {@link NDArray} from a given {@link RandomAccessibleInterval} and an array with its dimensions order.
     * 
     * @param tensor
     *        The INDArray containing the wanted data.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
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
     * Builds a {@link NDArray} from a unsigned byte-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#UBYTE}.
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
     * Builds a {@link NDArray} from a unsigned integer-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#INT}.
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
     * Builds a {@link NDArray} from a unsigned float-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#FLOAT}.
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
     * Builds a {@link NDArray} from a unsigned double-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#DOUBLE}.
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
