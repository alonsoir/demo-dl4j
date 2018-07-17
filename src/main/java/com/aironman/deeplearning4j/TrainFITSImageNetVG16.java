package com.aironman.deeplearning4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.slf4j.Logger;

import nom.tam.fits.Fits;
import nom.tam.fits.FitsException;
import nom.tam.fits.ImageData;
import nom.tam.fits.ImageHDU;
import nom.tam.image.ImageTiler;
import nom.tam.image.StandardImageTiler;

public class TrainFITSImageNetVG16 {

	private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(TrainFITSImageNetVG16.class);

	private static final long seed = 12345;
	public static final Random RAND_NUM_GEN = new Random(seed);
	public static final String[] ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS;
	public static ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();
	public static BalancedPathFilter PATH_FILTER = new BalancedPathFilter(RAND_NUM_GEN, ALLOWED_FORMATS,
			LABEL_GENERATOR_MAKER);
	private static final int EPOCH = 5;
	private static final int BATCH_SIZE = 16;
	private static final int TRAIN_SIZE = 85;
	private static final int NUM_POSSIBLE_LABELS = 2;
	private static final int SAVING_INTERVAL = 100;
	public static String DATA_PATH = "resources";
	public static final String TRAIN_FOLDER = DATA_PATH + "/train_both";
	public static final String TEST_FOLDER = DATA_PATH + "/test_both";
	private static final String SAVING_PATH = DATA_PATH + "/saved/modelIteration_";
	private static final String FREEZE_UNTIL_LAYER = "fc2";
	// path to dataTraining file
	private static final String DATA_URL = "https://www.dropbox.com/preview/Public/wetransfer-39ab61.zip";

	private static final String LOCAL_ZIP_DATA_PATH = "/Users/aironman/Dropbox/Public/wetransfer-39ab61.zip";

	private static final String LOCAL_EXPANDED_DATA_PATH = DATA_PATH + "/wetransfer-39ab61";

	public static void main(String[] args) throws IOException {
		/*
		 * ZooModel zooModel = new VGG16();
		 * LOGGER.debug("Start Downloading VGG16 model..."); ComputationGraph
		 * preTrainedNet = (ComputationGraph)
		 * zooModel.initPretrained(PretrainedType.IMAGENET);
		 * LOGGER.debug(preTrainedNet.summary());
		 * 
		 * LOGGER.debug("Start Downloading Data..."); useUnzippedLocalFile();
		 * LOGGER.debug("Data unzipped"); // Define the File Paths File trainData = new
		 * File(TRAIN_FOLDER); File testData = new File(TEST_FOLDER);
		 */
		readFitFile();
	}

	private static void readFitFile() {

		try {
			List<File> filesdebuglder = Files.walk(Paths.get(LOCAL_EXPANDED_DATA_PATH))
											 .filter(Files::isRegularFile)
											 // .filter(line -> line.getName(0).toString().contains(".FIT"))
											 .map(Path::toFile)
											 .collect(Collectors.toList());
			System.out.println("There are " + filesdebuglder.size() + " .FIT files in folder " + LOCAL_EXPANDED_DATA_PATH);
			int count = 1;
			for (File afile : filesdebuglder) {
				System.out.println("Doing something cool with file " + afile.getName() + " ...");
				Fits fitsFile = new Fits(afile);
				ImageHDU imageHDU = (ImageHDU) fitsFile.readHDU();
				StandardImageTiler tiler = imageHDU.getTiler();
				short[][] tmp = (short[][] ) tiler.getCompleteImage();
				System.out.println("tmp is " + tmp);
				short imgData = tmp[0][0];
				System.out.println("imgData is " + imgData );
				count ++;
				System.out.println("Done with the file " + afile.getName() + " ... " + count);
				fitsFile.close();
				
				/*
				f = new Fits(afile);
				ImageHDU hdu = (ImageHDU) f.getHDU(0);
				int[][] image = (int[][]) hdu.getKernel();
				System.out.println(image.length);
				ImageData imageData = (ImageData) hdu.getData();
				int[][] _imageData = (int[][]) imageData.getData();
				System.out.println();
				ImageTiler anotherTiler = hdu.getTiler();
				int[] corners = new int[] { 950, 950 };
				int[] lengths = new int[] { 100, 100 };
				// short[] center = (short[]) tiler.getTile({950, 950}, {100, 100});
				short[] center = (short[]) anotherTiler.getTile(corners, lengths);
				System.out.println();
				*/
				
				/*
				 * Reading only parts of an Image
				 * 
				 * When reading image data users may not want to read an entire array especially
				 * if the data is very large. An ImageTiler can be used to read in only a
				 * portion of an array. The user can specify a box (or a sequence of boxes)
				 * within the image and extract the desired subsets. ImageTilers can be used for
				 * any image. The library will try to only read the subsets requested if the
				 * FITS data is being read from an uncompressed file but in many cases it will
				 * need to read in the entire image before subsetting.
				 * 
				 * Suppose the image we retrieve above has 2000x2000 pixels, but we only want to
				 * see the innermost 100x100 pixels. This can be achieved with
				 **/
			}

		} catch (FitsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch(ClassCastException e) {
			e.printStackTrace();
		}

	}

	private static void useUnzippedLocalFile() {
		if (!new File(TRAIN_FOLDER).exists()) {
			new File(LOCAL_EXPANDED_DATA_PATH);
		}
	}
}
