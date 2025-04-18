package com.ultralytics.yolo;

import android.graphics.Bitmap;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ImageProcessing {

    static {
        System.loadLibrary("image_processing");
    }

    // Java-based preprocessing with normalization
    public static ByteBuffer bitmapToInputBuffer(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, Predictor.INPUT_SIZE, Predictor.INPUT_SIZE, true);
        ByteBuffer buffer = ByteBuffer.allocateDirect(Predictor.INPUT_SIZE * Predictor.INPUT_SIZE * 3 * 4);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[Predictor.INPUT_SIZE * Predictor.INPUT_SIZE];
        resized.getPixels(pixels, 0, Predictor.INPUT_SIZE, 0, 0, Predictor.INPUT_SIZE, Predictor.INPUT_SIZE);

        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f; // Normalize to [0,1]
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            buffer.putFloat(r);
            buffer.putFloat(g);
            buffer.putFloat(b);
        }
        return buffer;
    }

    // Native method (optional)
    public native void argb2yolo(int[] src, ByteBuffer dest, int width, int height);
}