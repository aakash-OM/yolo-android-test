package com.ultralytics.yolo.predict.detect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;

import com.ultralytics.yolo.ImageProcessing;
import com.ultralytics.yolo.models.LocalYoloModel;
import com.ultralytics.yolo.models.YoloModel;
import com.ultralytics.yolo.predict.Predictor;
import com.ultralytics.yolo.predict.DetectedObject;

import org.json.JSONArray;
import org.json.JSONException;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

public class TfliteDetector extends Predictor {
    private Interpreter interpreter;
    private Object[] inputArray;
    private Map<Integer, Object> outputMap;
    private List<String> labels;

    public TfliteDetector(Context context) {
        super(context);
    }

    @Override
    public void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception {
        if (yoloModel instanceof LocalYoloModel) {
            LocalYoloModel localModel = (LocalYoloModel) yoloModel;
            labels = loadLabels(context.getAssets(), localModel.metadataPath);
            interpreter = new Interpreter(loadModelFile(context, localModel.modelPath));
        }
    }

    @Override
    public JSONArray predict(Bitmap bitmap) throws JSONException {
        ByteBuffer inputBuffer = ImageProcessing.bitmapToInputBuffer(bitmap);
        inputArray = new Object[]{inputBuffer};
        outputMap = new HashMap<>();
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(8400 * 7 * 4);
        outputBuffer.order(java.nio.ByteOrder.nativeOrder());
        outputMap.put(0, outputBuffer);

        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        ArrayList<DetectedObject> detections = parseOutput(outputBuffer);
        return convertToJSON(detections);
    }

    private ArrayList<DetectedObject> parseOutput(ByteBuffer outputBuffer) {
        ArrayList<DetectedObject> results = new ArrayList<>();
        outputBuffer.rewind();
        for (int i = 0; i < 8400; i++) {
            float x = outputBuffer.getFloat();   // Center X
            float y = outputBuffer.getFloat();   // Center Y
            float w = outputBuffer.getFloat();   // Width
            float h = outputBuffer.getFloat();   // Height
            float conf = outputBuffer.getFloat(); // Confidence
            int classId = outputBuffer.getInt();  // Class ID

            if (conf > 0.5f && classId >= 0 && classId < labels.size()) {
                RectF box = new RectF(x - w / 2, y - h / 2, x + w / 2, y + h / 2);
                results.add(new DetectedObject(conf, box, classId, labels.get(classId)));
            }
        }
        return results;
    }

    private JSONArray convertToJSON(ArrayList<DetectedObject> detections) throws JSONException {
        JSONArray jsonArray = new JSONArray();
        for (DetectedObject obj : detections) {
            jsonArray.put(obj.toJSON());
        }
        return jsonArray;
    }

    private ByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
