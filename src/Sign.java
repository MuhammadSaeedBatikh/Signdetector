import com.leapmotion.leap.Bone;
import com.leapmotion.leap.Finger;
import com.leapmotion.leap.Hand;
import com.leapmotion.leap.Vector;
import org.apache.commons.lang3.ArrayUtils;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Muhammad on 17/07/2017.
 */
public class Sign {
    // static enum Group {BACKHAND, FOREHAND}

    float[] features;
    int[] classifiedFrames = new int[20];
    Integer[] arr = {26, 42, 25, 55, 22, 16, 15, 12};
    List<Integer> excludedList = new ArrayList<Integer>(Arrays.asList(arr));
    int classifiedClass;


    public Sign(Hand hand) {
        gatherDataFromHand(hand);
        excludeFeatures();
        classifySign();
    }


    public void gatherDataFromHand(Hand hand) {
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
    }

    public void classifySign() {
        try {
            this.classifiedClass = MyClassifier.classify(features);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static Vector changeBasis(Vector d, Vector r, Vector n, Vector v) {
        float[] c = {d.getX(), d.getY(), d.getZ(), r.getX(), r.getY(), r.getZ(), n.getX(), n.getY(), n.getZ()};
        // h = 1/det
        float h = 1 / (c[0] * (c[4] * c[8] - c[5] * c[7]) + c[1] * (c[5] * c[6] - c[3] * c[8]) + c[2] * (c[3] * c[7] - c[4] * c[6]));
        float[] inverseOfBasis = {

                h * (c[4] * c[8] - c[5] * c[7]), h * (c[2] * c[7] - c[1] * c[8]), h * (c[1] * c[5] - c[2] * c[4]),

                h * (c[5] * c[6] - c[3] * c[8]), h * (c[0] * c[8] - c[2] * c[6]), h * (c[2] * c[3] - c[0] * c[5]),

                h * (c[3] * c[7] - c[4] * c[6]), h * (c[1] * c[6] - c[0] * c[7]), h * (c[0] * c[4] - c[1] * c[3])
        };
        Vector column1 = new Vector(inverseOfBasis[0], inverseOfBasis[1], inverseOfBasis[2]);
        Vector column2 = new Vector(inverseOfBasis[3], inverseOfBasis[4], inverseOfBasis[5]);
        Vector column3 = new Vector(inverseOfBasis[6], inverseOfBasis[7], inverseOfBasis[8]);
        //column1*x + column2*y +column3*z
        return column1.times(v.getX()).plus(column2.times(v.getY())).plus(column3.times(v.getZ()));
    }


    public void excludeFeatures() {
        for (int i = 0; i < 55; i++) {
            if (this.excludedList.contains(i)) {
                features = ArrayUtils.remove(features, i);
            }
        }
    }

}