import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:flutter/material.dart';

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.imageSize, this.results);
  final Size? imageSize;
  double? scaleX, scaleY;
  dynamic results;
  Face? face;
  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.greenAccent;
    for (String label in results.keys) {
      for (Face face in results[label]) {
        // face = results[label];
        scaleX = size.width / (imageSize?.width ?? 0);
        scaleY = size.height / (imageSize?.height ?? 0);
        canvas.drawRRect(
            _scaleRect(
                rect: face.boundingBox,
                imageSize: imageSize ?? const Size(0, 0),
                widgetSize: size,
                scaleX: scaleX ?? 0,
                scaleY: scaleY ?? 0),
            paint);
        TextSpan span = TextSpan(
            style: TextStyle(color: Colors.orange[300], fontSize: 15),
            text: label);
        TextPainter textPainter = TextPainter(
            text: span,
            textAlign: TextAlign.left,
            textDirection: TextDirection.ltr);
        textPainter.layout();
        textPainter.paint(
            canvas,
            Offset(
                size.width - (60 + face.boundingBox.left.toDouble()) * (scaleX ?? 0),
                (face.boundingBox.top.toDouble() - 10) * (scaleY ?? 0)));
      }
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.results != results;
  }
}

RRect _scaleRect(
    {required Rect rect,
    required Size imageSize,
    required Size widgetSize,
    required double scaleX,
    required double scaleY}) {
  return RRect.fromLTRBR(
      (widgetSize.width - rect.left.toDouble() * scaleX),
      rect.top.toDouble() * scaleY,
      widgetSize.width - rect.right.toDouble() * scaleX,
      rect.bottom.toDouble() * scaleY,
      const Radius.circular(10));
}