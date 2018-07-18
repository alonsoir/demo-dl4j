package com.aironman.deeplearning4j;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Random;
import java.util.zip.Adler32;

// import org.apache.ant.compress.taskdefs.Unzip;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

/***
 * This class is going to use converted jpg files from original FIT files.
 * Experimental.
 * 
 * @author aironman
 *
 */
public class TrainImage {

	private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(TrainImage.class);

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
	public static final String TRAIN_FOLDER = DATA_PATH + "/jpg_files/train_folder";
	public static final String TEST_FOLDER = DATA_PATH + "/jpg_files/test_folder";
	private static final String SAVING_PATH = "/Users/aironman/models/modelIteration_";
	private static final String FREEZE_UNTIL_LAYER = "fc2";

	public static void main(String[] args) throws IOException {
		ZooModel zooModel = new VGG16();
		System.out.println("Start Downloading VGG16 model...");
		ComputationGraph preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
		System.out.println(preTrainedNet.summary());

		// Define the File Paths
		File trainData = new File(TRAIN_FOLDER);
		File testData = new File(TEST_FOLDER);
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);
		FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);

		InputSplit[] sample = train.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
		DataSetIterator trainIterator = getDataSetIterator(sample[0]);
		DataSetIterator devIterator = getDataSetIterator(sample[1]);

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

		DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0]);
		int iEpoch = 0;
		int i = 0;
		while (iEpoch < EPOCH) {
			while (trainIterator.hasNext()) {
				DataSet trained = trainIterator.next();
				vgg16Transfer.fit(trained);
				File modelFile = new File(SAVING_PATH + i + "_epoch_" + iEpoch + ".zip");
				ModelSerializer.writeModel(vgg16Transfer, modelFile ,false);
				System.out.println("Model saved in " + SAVING_PATH + i + "_epoch_" + iEpoch + ".zip");
				evalOn(vgg16Transfer, devIterator, i);
				i++;
			}
			trainIterator.reset();
			iEpoch++;
			System.out.println("iEpoch is " + iEpoch);
			evalOn(vgg16Transfer, testIterator, iEpoch);
		}
		System.out.println("DONE main!");
	}

	public static void evalOn(ComputationGraph vgg16Transfer, DataSetIterator testIterator, int iEpoch)
			throws IOException {
		System.out.println("Evaluate model at iteration " + iEpoch + " ....");
		Evaluation eval = vgg16Transfer.evaluate(testIterator);
		System.out.println(eval.stats());
		testIterator.reset();
		System.out.println("DONE evalOn");

	}

	public static DataSetIterator getDataSetIterator(InputSplit sample) throws IOException {
		ImageRecordReader imageRecordReader = new ImageRecordReader(224, 224, 3, LABEL_GENERATOR_MAKER);
		imageRecordReader.initialize(sample);

		DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1,
				NUM_POSSIBLE_LABELS);
		iterator.setPreProcessor(new VGG16ImagePreProcessor());
		return iterator;
	}
}
