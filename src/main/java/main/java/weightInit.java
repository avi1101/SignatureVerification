package main.java;


import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class weightInit {
    public static void main(String[] args) throws Exception {
        MultiLayerConfiguration autoencoderConfig = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAGRAD)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(250).nOut(10)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(10).nOut(250)
                        .build())
                .layer(3, new OutputLayer.Builder().nIn(250).nOut(784)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork autoencoder = new MultiLayerNetwork(autoencoderConfig);
        autoencoder.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        DataSetIterator mnistIter = new MnistDataSetIterator(100, 50000, false);
        List<INDArray> inputFeaturesTrain = new ArrayList<INDArray>();
        List<INDArray> inputFeaturesTest = new ArrayList<INDArray>();
        List<INDArray> inputLabelsTest = new ArrayList<INDArray>();

        Random r = new Random(12345);
        while (mnistIter.hasNext()) {
            DataSet ds = mnistIter.next();
            // 80/20 split (from miniBatch = 100)
            SplitTestAndTrain split = ds.splitTestAndTrain(80, r);
            inputFeaturesTrain.add(split.getTrain().getFeatureMatrix());
            DataSet dsTest = split.getTest();
            inputFeaturesTest.add(dsTest.getFeatureMatrix());
            // Convert from one-hot representation -> index
            INDArray indexes = Nd4j.argMax(dsTest.getLabels(), 1);
            inputLabelsTest.add(indexes);
        }

        Layer autoencoderFirstlayer_preTrain = autoencoder.getLayer(0);
        INDArray preTrainweights = autoencoderFirstlayer_preTrain.params();
      //  DataBuffer dataBuffer = preTrainweights.data();
       // double[] array = dataBuffer.asDouble();

        // Train model:
        int nEpochs = 3;
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            for (INDArray data : inputFeaturesTrain) {
                autoencoder.fit(data, data);
            }
        }
        Layer autoencoderLayer_afterTrain = autoencoder.getLayer(0);
        INDArray trainWeights = autoencoderLayer_afterTrain.params();
      //  DataBuffer dataBuffer1 = trainWeights.data();

/**
 * this part adds uses the found weights to calculate a new autoencoder
 */

        MultiLayerConfiguration autoencoderTestConfig = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAGRAD)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(250).nOut(10)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(10).nOut(250)
                        .build())
                .layer(3, new OutputLayer.Builder().nIn(250).nOut(784)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork autoencoderTestingNet = new MultiLayerNetwork(autoencoderTestConfig);
        autoencoderTestingNet.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        Layer autoencoderTestingFirstLayer = autoencoderTestingNet.getLayer(0);
        autoencoderTestingFirstLayer.setParamsViewArray(trainWeights);
        //autoencoderTestingFirstLayer.setParams(trainWeights);
       // autoencoderTestingNet.setParams(trainWeights);


        mnistIter.reset();
        List<INDArray> autoencoderTestingFeaturesTrain = new ArrayList<INDArray>();
        List<INDArray> autoencoderTestingFeaturesTest = new ArrayList<INDArray>();
        List<INDArray> autoencoderTestingLabelsTest = new ArrayList<INDArray>();
        while (mnistIter.hasNext()) {
            DataSet ds2 = mnistIter.next();
            // 80/20 split (from miniBatch = 100)
            SplitTestAndTrain split2 = ds2.splitTestAndTrain(80, r);
            autoencoderTestingFeaturesTrain.add(split2.getTrain().getFeatureMatrix());
            DataSet dsTest2 = split2.getTest();
            autoencoderTestingFeaturesTest.add(dsTest2.getFeatureMatrix());
            // Convert from one-hot representation -> index
            INDArray indexes2 = Nd4j.argMax(dsTest2.getLabels(), 1);
            autoencoderTestingLabelsTest.add(indexes2);
        }

        // Train model:
       // int nEpochs2 = 3;
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            for (INDArray data : autoencoderTestingFeaturesTrain) {
                autoencoder.fit(data, data);
            }
            // System.out.println("Epoch " + epoch + " complete");
        }

            /**
             Layer lr1 = autoencoder.getLayer(0
             );
             INDArray arr1 = lr1.params();
             DataBuffer dataBuffer1 = arr1.data();
             double[] array1 = dataBuffer1.asDouble(); // or any type you want
             for(int i=0;i<1000;i++) {
             System.out.println(array1[i] - array[i]);
             System.out.println(arr.minNumber());
             System.out.println(" ");
             System.out.println("**************************************");
             System.out.println(" ");
             }
             **/
        for (int i = 0; i < nEpochs; i++) {
            //autoencoder.pretrain(mnistIter);
            System.out.format("Completed epoch %d", i);
            Evaluation eval = autoencoderTestingNet.evaluate(mnistIter);
            System.out.println((eval.stats()));
            mnistIter.reset();
            mnistIter.reset();
        }
        }
    }



