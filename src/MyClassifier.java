import org.apache.commons.lang3.ArrayUtils;
import weka.classifiers.lazy.IBk;
import weka.core.*;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Muhammad on 26/07/2017.
 */
public class MyClassifier {
   /* static String attributesNames =
            "F0distalX,F0distalY,F0distalZ,F0IntermediateX,F0IntermediateY,F0IntermediateZ,F0ProximalX,F0ProximalY,F0ProximalZ,F0TipDistanceToPalm," +
                    "F1distalX,F1distalY,F1IntermediateX,F1IntermediateY,F1ProximalX,F1ProximalY,F1ProximalZ,F1TipDistanceToPalm," +
                    "F2distalX,F2distalY,F2IntermediateX,F2IntermediateY,F2ProximalY,F2ProximalZ,F2TipDistanceToPalm," +
                    "F3distalX,F3distalY,F3IntermediateX,F3IntermediateY,F3IntermediateZ,F3ProximalX,F3ProximalY,F3ProximalZ,F3TipDistanceToPalm," +
                    "F4distalX,F4distalY,F4IntermediateX,F4IntermediateY,F4IntermediateZ,F4ProximalX,F4ProximalY,F4ProximalZ,F4TipDistanceToPalm," +
                    "grabAngle,pinchDistance,armPitch,armYaw"; */  //features excluded
    static String attributesNames =
            "F0distalX,F0distalY,F0distalZ,F0IntermediateX,F0IntermediateY,F0IntermediateZ,F0ProximalX,F0ProximalY,F0ProximalZ,F0TipDistanceToPalm," +
            "F1distalX,F1distalY,F1distalZ,F1IntermediateX,F1IntermediateY,F1IntermediateZ,F1ProximalX,F1ProximalY,F1ProximalZ,F1TipDistanceToPalm," +
            "F2distalX,F2distalY,F2distalZ,F2IntermediateX,F2IntermediateY,F2IntermediateZ,F2ProximalX,F2ProximalY,F2ProximalZ,F2TipDistanceToPalm," +
            "F3distalX,F3distalY,F3distalZ,F3IntermediateX,F3IntermediateY,F3IntermediateZ,F3ProximalX,F3ProximalY,F3ProximalZ,F3TipDistanceToPalm," +
            "F4distalX,F4distalY,F4distalZ,F4IntermediateX,F4IntermediateY,F4IntermediateZ,F4ProximalX,F4ProximalY,F4ProximalZ,F4TipDistanceToPalm," +
            "grabAngle,pinchDistance,armPitch,armYaw,armRoll";
    static String[] attributesNamesArray = attributesNames.split(",");
    static ArrayList<Attribute> attributes = new ArrayList<>();
    static String classesString = "alph,baa,taa,thaa,geem,hhaa,khaa,daal,tzaal,raa,zaay,seen,sheen,saad,daad,ttaa,zaa,ayin,gheen,faa,qaaf,kaaf,laam,meem,noon,haa,wau,yaa";
    static String[] classesArray = classesString.split(",");
    static IBk knnClassifier;
    static Instances dataSet;


    public static int classify(String record) throws Exception {
        String[] frameArr = record.split(",");
        double[] inputAtt = Arrays.stream(frameArr).map(s -> Float.parseFloat(s)).mapToDouble(Float::floatValue).toArray();
        return classify(inputAtt);
    }

    public static int classify(double[] record) throws Exception {
        Instance instance = new DenseInstance(dataSet.numAttributes());
        dataSet.add(instance);
        instance.setDataset(dataSet);

        for (int i = 0; i < record.length; i++) {
            instance.setValue(attributes.get(i), record[i]);
        }
        double d = knnClassifier.classifyInstance(instance);

        return (int) d;
    }

    public static int classify(float[] record) throws Exception {
        return classify(IntStream.range(0, record.length).mapToDouble(i -> record[i]).toArray());
    }

    /*public static String classify(double[][] recordsTable) throws Exception {
        return classify(reduceToSingleRecord(recordsTable));
    }*/

    public static String classify(double[][] recordsTable) throws Exception {
        int[] classifiedClass = new int[recordsTable.length];
        for (int i = 0; i < recordsTable.length; i++) {
            classifiedClass[i] = classify(recordsTable[i]);
        }
        int index = getHighestOccurrenceIndex(classifiedClass);
        return classesArray[index];
    }

    public static int classify(String[] inputBuffer) throws Exception {
        return classify(reduceToSingleRecord(inputBuffer));
    }

    public static void initializeClassifier(String modelPath) throws Exception {
        for (int i = 0; i < attributesNamesArray.length; i++) {
            attributes.add(new Attribute(attributesNamesArray[i]));
        }

        attributes.add(new Attribute("@@class@@", (List<String>) Arrays.asList(classesArray)));
        dataSet = new Instances("TextInstances", attributes, 0);
        dataSet.setClassIndex(dataSet.numAttributes() - 1);
        knnClassifier = (IBk) SerializationHelper.read(new FileInputStream(modelPath));

    }

    public static double[] reduceToSingleRecord(String[] records) {

        double[] singleRecord = Arrays.stream(records[0].split(",")).map(s -> Float.parseFloat(s)).mapToDouble(Float::floatValue).toArray();
        double[][] recordsTable = new double[records.length][singleRecord.length];
        for (int i = 1; i < records.length; i++) {
            recordsTable[i] = Arrays.stream(records[i].split(",")).map(s -> Float.parseFloat(s)).mapToDouble(Float::floatValue).toArray();
        }
        return reduceToSingleRecord(recordsTable);
    }

    public static double[] reduceToSingleRecord(double[][] recordsTable) {
        double[] result = new double[recordsTable[0].length];
        for (int i = 0; i < recordsTable[0].length; i++) {
            for (int j = 0; j < recordsTable.length; j++) {
                result[i] += recordsTable[j][i];
            }
            result[i] /= recordsTable.length;
        }
        return result;
    }

    public static int getHighestOccurrenceIndex(int[] arr) {
        Arrays.sort(arr);

        int previous = arr[0];
        int popular = arr[0];
        int count = 1;
        int maxCount = 1;

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] == previous)
                count++;
            else {
                if (count > maxCount) {
                    popular = arr[i - 1];
                    maxCount = count;
                }
                previous = arr[i];
                count = 1;
            }
        }
        return count > maxCount ? arr[arr.length - 1] : popular;
    }
}
