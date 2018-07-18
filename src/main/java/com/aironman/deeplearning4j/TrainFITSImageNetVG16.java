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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import nom.tam.fits.Fits;
import nom.tam.fits.FitsException;
import nom.tam.fits.ImageData;
import nom.tam.fits.ImageHDU;
import nom.tam.image.ImageTiler;
import nom.tam.image.StandardImageTiler;

/***
 * This class want to use the NASA library to manipulate FIT files. To future research.
 * 
 * @author aironman
 *
 */
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

		ZooModel zooModel = new VGG16();
		System.out.println("Start Downloading VGG16 model...");
		ComputationGraph preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
		System.out.println();
		System.out.println(preTrainedNet.summary());

		readFitFiles();

		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder().learningRate(5e-5)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
				.seed(seed).build();

		ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
				.fineTuneConfiguration(fineTuneConf).setFeatureExtractor(FREEZE_UNTIL_LAYER)
				.removeVertexKeepConnections("predictions")
				.addLayer("predictions",
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(4096)
								.nOut(NUM_POSSIBLE_LABELS).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
								.build(),
						FREEZE_UNTIL_LAYER)
				.build();
		vgg16Transfer.setListeners(new ScoreIterationListener(5));
		System.out.println(vgg16Transfer.summary());
		System.out.println("DONE main!");
	}

	private static void readFitFiles() {

		int count = 1;
		try {
			/*
			List<File> filesdebuglder = null;
			*/
			List<File> filesdebuglder = Files.walk(Paths.get(LOCAL_EXPANDED_DATA_PATH)).filter(Files::isRegularFile)
					// .filter(line -> line.getName(0).toString().contains(".FIT"))
					.map(Path::toFile).collect(Collectors.toList());
			
			System.out.println(
					"There are " + filesdebuglder.size() + " .FIT files in folder " + LOCAL_EXPANDED_DATA_PATH);
			
			for (File afile : filesdebuglder) {
				System.out.println("Doing something cool with file " + afile.getName() + " ...");
				Fits fitsFile = new Fits(afile);
				ImageHDU imageHDU = (ImageHDU) fitsFile.readHDU();
				StandardImageTiler tiler = imageHDU.getTiler();
				short[][] tmp = (short[][]) tiler.getCompleteImage();
				System.out.println("tmp.length: " + tmp.length);
				short imgData = tmp[0][0];
				System.out.println("imgData is " + imgData);
				int[] corners = new int[] { 0, 0 };
				int[] lengths = new int[] { 100, 100 };
				// what the fuck this method returns Object????
				short[] center = (short[]) tiler.getTile(corners, lengths);
				System.out.println("center is " + center[0] + " center.length: " + center.length);
				count++;
				System.out.println("Done with the file " + afile.getName() + " ... " + count);
				fitsFile.close();
			}

		} catch (FitsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassCastException e) {
			e.printStackTrace();
		}
		System.out.println("DONE readFitFiles!");
	}

	private static void useUnzippedLocalFile() {
		if (!new File(TRAIN_FOLDER).exists()) {
			new File(LOCAL_EXPANDED_DATA_PATH);
		}
	}
}
