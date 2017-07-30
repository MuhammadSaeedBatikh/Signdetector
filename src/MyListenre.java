import com.leapmotion.leap.*;
import com.sun.deploy.util.ArrayUtil;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import weka.classifiers.Classifier;

class MyListener extends Listener {
    long frameCounter = 0;
    long prevTime = System.currentTimeMillis();
    int resultsCounter = 0;
    String classFeature;
    FileWriter trainingSet;
    float[] features;
    int[] classifiedFrames = new int[20];
    int currFrame = 0;
    String outputFileName = "ranking training set.csv";
    Integer[] arr = {26, 42, 25, 55, 22, 16, 15, 12};
    List<Integer> excludedList = new ArrayList<Integer>(Arrays.asList(arr));
    Sign sign;
    static String attributesNames =
            "F0distalX,F0distalY,F0distalZ,F0IntermediateX,F0IntermediateY,F0IntermediateZ,F0ProximalX,F0ProximalY,F0ProximalZ,F0TipDistanceToPalm," +
                    "F1distalX,F1distalY,F1IntermediateX,F1IntermediateY,F1ProximalX,F1ProximalY,F1ProximalZ,F1TipDistanceToPalm," +
                    "F2distalX,F2distalY,F2IntermediateX,F2IntermediateY,F2ProximalY,F2ProximalZ,F2TipDistanceToPalm," +
                    "F3distalX,F3distalY,F3IntermediateX,F3IntermediateY,F3IntermediateZ,F3ProximalX,F3ProximalY,F3ProximalZ,F3TipDistanceToPalm," +
                    "F4distalX,F4distalY,F4IntermediateX,F4IntermediateY,F4IntermediateZ,F4ProximalX,F4ProximalY,F4ProximalZ,F4TipDistanceToPalm," +
                    "grabAngle,pinchDistance,armPitch,armYaw";//features excluded

    /*String attributesNames =
            "F0distalX,F0distalY,F0distalZ,F0IntermediateX,F0IntermediateY,F0IntermediateZ,F0ProximalX,F0ProximalY,F0ProximalZ,F0TipDistanceToPalm," +
                    "F1distalX,F1distalY,F1distalZ,F1IntermediateX,F1IntermediateY,F1IntermediateZ,F1ProximalX,F1ProximalY,F1ProximalZ,F1TipDistanceToPalm," +
                    "F2distalX,F2distalY,F2distalZ,F2IntermediateX,F2IntermediateY,F2IntermediateZ,F2ProximalX,F2ProximalY,F2ProximalZ,F2TipDistanceToPalm," +
                    "F3distalX,F3distalY,F3distalZ,F3IntermediateX,F3IntermediateY,F3IntermediateZ,F3ProximalX,F3ProximalY,F3ProximalZ,F3TipDistanceToPalm," +
                    "F4distalX,F4distalY,F4distalZ,F4IntermediateX,F4IntermediateY,F4IntermediateZ,F4ProximalX,F4ProximalY,F4ProximalZ,F4TipDistanceToPalm," +
                    "grabAngle,pinchDistance,armPitch,armYaw,armRoll,class";*/ //full set of features

    public void initialize(String classFeature) throws IOException {
        this.classFeature = classFeature;
        trainingSet = new FileWriter(new File(outputFileName), true);
        // save(this.attributesNames+"\n");
    }

    @Override
    public void onConnect(Controller controller) {
        System.out.println("connected");
    }

    @Override
    public void onDisconnect(Controller controller) {
        System.out.println("disconnected");
    }

    @Override
    public void onFrame(Controller controller) {
        {
            long currTime = System.currentTimeMillis();
            Frame frame = controller.frame();
            if (!controller.frame().hands().isEmpty()) {
                if (currTime - prevTime > 100) {
                    Hand hand = frame.hands().rightmost();
                    {
                        int count = hand.fingers().count();
                        Vector d = hand.direction();
                        Vector n = hand.palmNormal();
                        Vector r = d.cross(n).normalized();
                        this.features = new float[55];
                        int j = 0;
                        int i;
                        for (i = 0; i < count; i++) {
                            Finger finger = hand.fingers().get(i);
                            Vector tipPosition = finger.bone(Bone.Type.TYPE_DISTAL).center();
                            Vector palmPosition = hand.palmPosition();
                            float[] distalPhalangeDirection = Sign.changeBasis(d, r, n, finger.bone(Bone.Type.TYPE_DISTAL).direction()).toFloatArray();      //tips directions
                            System.arraycopy(distalPhalangeDirection, 0, features, j, 3);
                            j += 3;
                            float[] intermediatePhalangeDirection = Sign.changeBasis(d, r, n, finger.bone(Bone.Type.TYPE_INTERMEDIATE).direction()).toFloatArray();      //tips directions
                            System.arraycopy(intermediatePhalangeDirection, 0, features, j, 3);
                            j += 3;
                            float[] proximalPhalangeDirectios = Sign.changeBasis(d, r, n, finger.bone(Bone.Type.TYPE_PROXIMAL).direction()).toFloatArray();
                            System.arraycopy(proximalPhalangeDirectios, 0, features, j, 3);
                            float distance = palmPosition.distanceTo(tipPosition);
                            j += 3;
                            features[j++] = distance;
                        }

                        features[j] = hand.grabAngle();
                        features[++j] = hand.pinchDistance();
                        Vector armDirection = Sign.changeBasis(d, r, n, hand.arm().direction());
                        float[] armAngles = {armDirection.pitch(), armDirection.yaw(), armDirection.roll()};
                        System.arraycopy(armAngles, 0, features, ++j, 3);
                        j += 3;
                       // excludeFeatures();
                        try {
                           classifiedFrames[currFrame]= MyClassifier.classify(features);
                           currFrame++;
                        } catch (Exception e) {
                            System.err.println("error : mostly because mismatched number of features");
                        }
                    }
                    if (currFrame == 19) {
                        int max = MyClassifier.getHighestOccurrenceIndex(classifiedFrames);
                        System.out.println(MyClassifier.classesArray[max]);
                        currFrame = 0;
                    }
                }
            }
        }
    }

    public void save(String st) {
        try {
            trainingSet.write(st);
            trainingSet.close();
            trainingSet = new FileWriter(new File(outputFileName), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String arrayToCSV(float[] array) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < array.length - 1; i++) {
            stringBuilder.append(array[i]).append(",");
        }
        stringBuilder.append(array[array.length - 1]);
        return stringBuilder.toString();
    }


    public String featuresArrayToCSV() {
        StringBuilder stringBuilder = new StringBuilder();
        int length = this.features.length;
        for (int i = 0; i < length - 1; i++) {
            stringBuilder.append(this.features[i]).append(",");
        }
        stringBuilder.append(this.features[length - 1]);
        return stringBuilder.toString();

    }


    public void save() throws IOException {
        trainingSet.close();
        trainingSet = new FileWriter(new File(outputFileName), true);
    }


    public void setClassFeature(String classFeature) {
        this.classFeature = classFeature;
    }

    public void excludeFeatures() {
        for (int i = 0; i < 55; i++) {
            if (this.excludedList.contains(i)) {
                features = ArrayUtils.remove(features, i);
            }
        }
    }

}
