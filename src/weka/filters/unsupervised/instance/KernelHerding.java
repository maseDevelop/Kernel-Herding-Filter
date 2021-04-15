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

            ArrayList newIstanceIndex = new ArrayList<Integer>();

            double currentSum = 0.0;
            double percentOfDataset = 0.0;
            double subsetSize = 0.0;
            int currentMaxIndex = 0;
            double currentMaxValue = 0.0;
            double scalar = 0.0;
            double outputVal;

            m_Kernel.buildKernel(instances); //Building the kernel with the input dataset

            //System.out.println(m_Kernel.eval(0,0,instances.instance(0)));

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

                //Subtracting the values of evaling an instance by and instance i.e. instance 0 and instance 0
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

            /*System.out.println(instances.toSummaryString());
            System.out.println("Dataset size: " + instances.size());
            System.out.println("Percent of dataset: " + percentOfDataset);
            System.out.println("SubsetSize: " + subsetSize);
            System.out.println("SubsetSize(int) : " + (int)subsetSize);*/

            //Getting initial value for subset array - FIRST INSTANCE in array
            newIstanceIndex.add(currentMaxIndex);


            //Setting current sum for part 2 of the equation
            //currentSum = currentMaxValue;

            //Creating new instance to feed return
            Instances newInstances = new Instances(instances,currentMaxIndex,1);

            /*for (int i = 0; i < newInstances.size(); i++) {
                System.out.println("subset: " + newInstances.instance(i));
            }*/
            //System.out.println("-----------------------------------------");

            //int count = 2; //Already add an instance before the loop is iterated over
            int count = 2; //Already add an instance before the loop is iterated over
            ArrayList sortedMax = new ArrayList<Integer>();
            //Building subset with Kernel Herding Equation starting at subset -1 as already added first element
            while(count <= (int)subsetSize) {//Might not work when 8.88%

                currentMaxIndex = 0;
                currentMaxValue = 0;
                sortedMax.clear();

                //Going through each element in the array
                for (int j = 0; j < similarityToOtherArray.length; j++) {

                        //System.out.println("size" + (newIstanceIndex.size()-1));
                        //System.out.println("similarity to other: " + similarityToOtherSamples[j]);

                        //similarityToOtherSamples[j] += m_Kernel.eval(j,(int)newIstanceIndex.get(newIstanceIndex.size()-1),instances.instance(j));
                        similarityToOtherSamples[j] += m_Kernel.eval(j, (int) newIstanceIndex.get(newIstanceIndex.size() - 1), instances.instance(j));



                        //System.out.println("K: " +m_Kernel.eval(j,(int)newIstanceIndex.get(newIstanceIndex.size()-1),instances.instance(j)));

                        //System.out.println("similarity to other2: " + similarityToOtherSamples[j]);
                        //System.out.println("value kernel: " + m_Kernel.eval(j,(int)newIstanceIndex.get(newIstanceIndex.size()-1),instances.instance(j)));

                        //Taking the average of the second part of the function
                        //scalar = similarityToOtherSamples[j]/(newIstanceIndex.size() + 1);
                        scalar = similarityToOtherSamples[j] / (newIstanceIndex.size() + 1);

                        //Executing the full function
                        currentSum = similarityToOtherArray[j] - scalar;

                        //Checking if it is the max value
                        if (currentSum > currentMaxValue) {
                            sortedMax.add(j);
                            currentMaxValue = currentSum;
                            currentMaxIndex = j;//Setting index of the current max value
                        }
                    }

                if(newIstanceIndex.contains(currentMaxIndex)){
                    int c = sortedMax.size() -2;//Second best Index to choose from
                    boolean foundOption = false;
                    while(!foundOption){
                        if(c <= 0){
                            //Choose a random sample because all the indexs found seem to be in the list already
                            Random rand = new Random();
                            int newIndex = rand.nextInt(instances.size());;

                            while(newIstanceIndex.contains(newIndex)){
                                newIndex = rand.nextInt(instances.size());
                            }
                            currentMaxIndex = newIndex;
                            foundOption = true;
                        }
                        else if(!newIstanceIndex.contains(sortedMax.get(c))){
                            currentMaxIndex = (Integer) sortedMax.get(c);
                            foundOption = true;
                        }
                        c--;//Moving down the index array
                    }
                }




                newIstanceIndex.add(currentMaxIndex); //Adding the max value to subset array
                newInstances.add(instances.instance(currentMaxIndex));

                count++;


            }

            /*System.out.println("------------------------------");
            System.out.println("Count: " + count);
            for (int i = 0; i < newInstances.size(); i++) {
                System.out.println(newIstanceIndex.get(i));
            }
            System.out.println("size of new subset: " + newInstances.size());
            System.out.println("------------------------------");*/



            //m_Kernel.clean();//Should I do this
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