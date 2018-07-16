package com.aironman.deeplearning4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.slf4j.Logger;

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
		LOGGER.info("Start Downloading VGG16 model...");
		ComputationGraph preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
		LOGGER.info(preTrainedNet.summary());

		LOGGER.info("Start Downloading Data...");
		useUnzippedLocalFile();
		LOGGER.info("Data unzipped");
		// Define the File Paths
		File trainData = new File(TRAIN_FOLDER);
		File testData = new File(TEST_FOLDER);
	}

	private static void useUnzippedLocalFile() {
		if (!new File(TRAIN_FOLDER).exists()) {
			new File(LOCAL_EXPANDED_DATA_PATH);
		}
	}
}
