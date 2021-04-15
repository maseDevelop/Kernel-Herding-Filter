package weka.filters.unsupervised.instance;

import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.filters.SimpleBatchFilter;

import java.util.*;

public class KernelHerding extends SimpleBatchFilter {

    /** for serialization */
    static final long serialVersionUID = -251831442047263433L;

    /** The kernel function to use. */
    protected Kernel m_Kernel = new PolyKernel();

    /** The subsample size, percent of original set, default 100% */
    protected double m_SampleSizePercent = 100;

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = getKernel().getCapabilities();
        result.setOwner(this);
        result.setMinimumNumberInstances(0);
        result.enableAllClasses();
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);

        return result;
    }

    /** Handling the kernel parameter. */
    @OptionMetadata(
            displayName = "Kernel function",
            description = "The kernel function to use.", displayOrder = 1,
            commandLineParamName = "K",
            commandLineParamSynopsis = "-K <kernel specification>")
    public Kernel getKernel() {  return m_Kernel; }
    public void setKernel(Kernel value) { m_Kernel = value; }

    /** Handling the parameter setting the sample size. */
    @OptionMetadata(
            displayName = "Percentage of the training set to sample.",
            description = "The percentage of the training set to sample (between 0 and 100).", displayOrder = 3,
            commandLineParamName = "Z",
            commandLineParamSynopsis = "-Z <double>")
    public void setSampleSizePercent(double newSampleSizePercent) { m_SampleSizePercent = newSampleSizePercent; }
    public double getSampleSizePercent() { return m_SampleSizePercent; }

    @Override
    public String globalInfo() { return "A filter implementing kernel herding for unsupervised subsampling of data."; }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        return new Instances(inputFormat, 0);
    }

    @Override
    protected Instances process(Instances instances) throws Exception {

        // We only modify the first batch of data that this filter receives
        // (e.g., the training data of a classifier, not the test data)
        if (!isFirstBatchDone()) {

            //Process expected values of each instance with respect to every other instance
            double[] similarityToOtherArray = new double[instances.size()];
            double[] similarityToOtherSamples = new double[instances.size()];

            ArrayList newInstanceIndex = new ArrayList<Integer>();

            double currentSum = 0.0;
            double percentOfDataset = 0.0;
            double subsetSize = 0.0;
            int currentMaxIndex = 0;
            double currentMaxValue = 0.0;
            double scalar = 0.0;
            double outputVal;

            m_Kernel.buildKernel(instances); //Building the kernel with the input dataset

            //Going through each value in the dataset
            for (int i = 0; i < instances.size(); i++) {

                //Initialising array with values
                similarityToOtherSamples[i] = 0.0;
                //resetting output value
                outputVal = 0;

                //Comparing it to all other values in the dataset
                for (int j = 0; j < instances.size(); j++) {
                    //Calling kernel function and storing scalar output
                    outputVal += m_Kernel.eval(i,j,instances.instance(i));
                }

                //Subtracting the values of evaluating an instance by and instance i.e. instance 0 and instance 0
                outputVal -= 1.0;

                //Getting the expected value of the sum and storing it
                similarityToOtherArray[i] = outputVal/instances.size();

                //Finding max value to start subset with
                if(similarityToOtherArray[i] > currentMaxValue){
                    currentMaxValue = similarityToOtherArray[i];
                    currentMaxIndex = i;//Setting index of the current max value
                }
            }


            //Getting the sub sample size.
            percentOfDataset = m_SampleSizePercent / 100;
            subsetSize = (instances.size() * percentOfDataset);

            //Getting initial value for subset array - FIRST INSTANCE in array
            newInstanceIndex.add(currentMaxIndex);

            //Creating new instance to feed return
            Instances newInstances = new Instances(instances,currentMaxIndex,1);

            int count = 2; //Already add an instance before the loop is iterated over
            ArrayList sortedMax = new ArrayList<Integer>();
            //Building subset with Kernel Herding Equation starting at subset -1 as already added first element
            while(count <= (int)subsetSize) {//If subsetSize is 8.1 it will create subset of size 8

                currentMaxIndex = 0;
                currentMaxValue = 0;
                sortedMax.clear();

                //Going through each element in the array
                for (int j = 0; j < similarityToOtherArray.length; j++) {

                        similarityToOtherSamples[j] += m_Kernel.eval(j, (int) newInstanceIndex.get(newInstanceIndex.size() - 1), instances.instance(j));

                        //Taking the average of the second part of the function
                        scalar = similarityToOtherSamples[j] / (newInstanceIndex.size() + 1);
                        //Executing the full function
                        currentSum = similarityToOtherArray[j] - scalar;

                        //Checking if it is the max value
                        if (currentSum > currentMaxValue) {
                            sortedMax.add(j);
                            currentMaxValue = currentSum;
                            currentMaxIndex = j;//Setting index of the current max value
                        }
                    }

                if(newInstanceIndex.contains(currentMaxIndex)){
                    int c = sortedMax.size() -2;//Second best Index to choose from
                    boolean foundOption = false;
                    while(!foundOption){
                        if(c <= 0){
                            //Choose a random sample because all the indexs found seem to be in the list already
                            Random rand = new Random();
                            int newIndex = rand.nextInt(instances.size());;

                            //Make sure that it is not in the new subset
                            while(newInstanceIndex.contains(newIndex)){
                                newIndex = rand.nextInt(instances.size());
                            }
                            currentMaxIndex = newIndex;
                            foundOption = true;
                        }
                        else if(!newInstanceIndex.contains(sortedMax.get(c))){
                            currentMaxIndex = (Integer) sortedMax.get(c);
                            foundOption = true;
                        }
                        c--;//Moving down the index array
                    }
                }

                newInstanceIndex.add(currentMaxIndex);
                newInstances.add(instances.instance(currentMaxIndex));
                count++;
            }

            //m_Kernel.clean();
            //this.m_FirstBatchDone = true;
            return newInstances;

        }

        return instances;
    }

    /**
     * The main method used for running this filter from the command-line interface.
     *
     * @param options the command-line options
     */
    public static void main(String[] options) {
        runFilter(new KernelHerding(), options);
    }
}