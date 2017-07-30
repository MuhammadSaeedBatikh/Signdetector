import com.leapmotion.leap.Vector;

/**
 * Created by Muhammad on 24/07/2017.
 */
public class MyVector {
    public float x;
    public float y;
    public float z;

    public MyVector(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public MyVector(Vector v) {
        this.x = v.getX();
        this.y = v.getY();
        this.z = v.getZ();
    }

    public MyVector cross(MyVector v) {
        float bx = v.x;
        float by = v.y;
        float bz = v.z;
        float x = this.y * bz - this.z * by;
        float y = this.z * bx - this.x * bz;
        float z = this.x * by - this.y * bx;
        MyVector product = new MyVector(x, y, z);
        return product;
    }

    public MyVector plus(MyVector v) {
        return new MyVector(x + v.x, y + v.y, z + v.z);
    }

    public MyVector minus(MyVector v) {
        return new MyVector(x - v.x, y - v.y, z - v.z);

    }

    public float dot(MyVector v) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }

    public float magnitude() {
        return (float) Math.sqrt(x * x + y * y + z * z);
    }

    public MyVector divide(float c) {
        if (c != 0) {
            float a = 1 / c;
            return new MyVector(a * x, a * y, a * z);
        } else
            return new MyVector(Float.NaN, Float.NaN, Float.NaN);
    }

    public MyVector normalized() {
        if (magnitude() != 0) {
            return this.divide(magnitude());
        } else
            return new MyVector(0, 0, 0);
    }

    public float angleTo(MyVector v) {
        if (v.magnitude() != 0 & this.magnitude() != 0) {
            return (float) Math.acos(this.dot(v) / (this.magnitude() * v.magnitude()));
        }
        return Float.NaN;
    }


    public float distanceTo(MyVector v) {
        return (float) Math.sqrt(Math.pow(v.x - this.x, 2) + Math.pow(v.y - this.y, 2) + Math.pow(v.z - this.z, 2));
    }

    public MyVector times(float a) {
        return new MyVector(a * x, a * y, a * z);
    }

    public static MyVector cross(MyVector a, MyVector b) {
        float ax = a.x;
        float ay = a.y;
        float az = a.z;
        float bx = b.x;
        float by = b.y;
        float bz = b.z;
        float x = ay * bz - az * by;
        float y = az * bx - ax * bz;
        float z = ax * by - ay * bx;
        MyVector product = new MyVector(x, y, z);
        return product;
    }

    public float getX() {
        return x;
    }

    public float getY() {
        return y;
    }

    public float getZ() {
        return z;
    }

    @Override
    public String toString() {
        return "(" + this.x + ", " + this.y + ", " + this.z + ")";
    }

    public String toCSV() {
        return this.x + "," + this.y + "," + this.z;
    }

    public float[] toFloatArray() {
        return new float[]{this.x, this.y, this.z};
    }

    public boolean equals(MyVector v) {
        return v.x == this.x & v.y == this.y & v.z == this.z;
    }
}
