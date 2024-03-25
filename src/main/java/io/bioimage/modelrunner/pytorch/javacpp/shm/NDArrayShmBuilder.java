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

package io.bioimage.modelrunner.pytorch.javacpp.shm;

import io.bioimage.modelrunner.pytorch.javacpp.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import net.imglib2.util.Cast;

import java.io.IOException;

/**
 * A helper class to build {@link SharedMemoryArray} from {@link org.bytedeco.pytorch.Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class NDArrayShmBuilder {

	
	/**
	 * Build a shared memory segment from a Pytorch tensor
	 * @param tensor
	 * 	the Pytorch tensor created using JavaCPP
	 * @param memoryName
	 * 	the shared memory region name
	 * @return the {@link SharedMemoryArray} object created
	 * @throws IOException if there is any error creating the shared memory segment
	 */
	public static SharedMemoryArray buildShma(org.bytedeco.pytorch.Tensor tensor, String memoryName) throws IOException {
		return SharedMemoryArray.createSHMAFromRAI(memoryName, Cast.unchecked(ImgLib2Builder.build(tensor)), false, true);
	}
}
