import 'dart:convert';
import 'dart:io';
import 'package:face_rec/detector_painter.dart';
import 'package:face_rec/utils.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:quiver/collection.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as imglib;

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  late File jsonFile;
  dynamic _scanResults;
  late CameraController _camera;
  late Interpreter interpreter;
  CameraLensDirection direction = CameraLensDirection.front;
  bool isDetecting = false;
  dynamic data = {};
  double threshold = 1.0;
  Directory? tempDir;
  List e1 = [];
  bool faceFound = false;
  final TextEditingController _name = TextEditingController();
  bool _isCameraInitialized = false;

  final FaceDetector _faceDetector = FaceDetector(options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate));

  final Map<DeviceOrientation, int> _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  @override
  void initState() {
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    _initializeCamera();
    super.initState();
  }

  Future<void> _loadModel() async {
    try {
      final gpuDelegateV2 = GpuDelegateV2(options: GpuDelegateOptionsV2());
      var interpreterOptions = InterpreterOptions()..addDelegate(gpuDelegateV2);
      interpreter = await Interpreter.fromAsset('assets/mobilefacenet.tflite', options: interpreterOptions);
    } on Exception {
      debugPrint('Error loading model');
    }
  }

  InputImage? _inputImageFromCameraImage(CameraImage image) {
    // get image rotation
    // it is used in android to convert the InputImage from Dart to Java
    // `rotation` is not used in iOS to convert the InputImage from Dart to Obj-C
    // in both platforms `rotation` and `camera.lensDirection` can be used to compensate `x` and `y` coordinates on a canvas
    final camera = _camera.description;
    final sensorOrientation = camera.sensorOrientation;
    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      var rotationCompensation = _orientations[_camera.value.deviceOrientation];
      if (rotationCompensation == null) return null;
      if (camera.lensDirection == CameraLensDirection.front) {
        // front-facing
        rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
      } else {
        // back-facing
        rotationCompensation =
            (sensorOrientation - rotationCompensation + 360) % 360;
      }
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }
    if (rotation == null) return null;

    // get image format
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    // validate format depending on platform
    // only supported formats:
    // * nv21 for Android
    // * bgra8888 for iOS
    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) return null;

    // since format is constraint to nv21 or bgra8888, both only have one plane
    if (image.planes.length != 1) return null;
    final plane = image.planes.first;

    // compose InputImage using bytes
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation, // used only in Android
        format: format, // used only in iOS
        bytesPerRow: plane.bytesPerRow, // used only in iOS
      ),
    );
  }

  Future<CameraDescription> _getCameraDescription() async {
    List<CameraDescription> cameras = await availableCameras();
    return cameras.firstWhere((CameraDescription camera) =>
        camera.lensDirection == CameraLensDirection.front);
  }

  _initializeCamera() async {
    await _loadModel();
    CameraDescription camera = await _getCameraDescription();
    _camera = CameraController(camera, ResolutionPreset.high, enableAudio: false, imageFormatGroup: ImageFormatGroup.nv21);
    InputImageRotation rotation = rotationIntToImageRotation(camera.sensorOrientation);

    try {
      await _camera.initialize();
      setState(() {
        _isCameraInitialized = true;
      });
      await Future.delayed(const Duration(milliseconds: 500));
      tempDir = await getApplicationDocumentsDirectory();
      String embPath = '${tempDir!.path}/emb.json';
      jsonFile = File(embPath);
      if (jsonFile.existsSync()) {
        data = json.decode(jsonFile.readAsStringSync());
      }
      _streamCamera(rotation, camera);
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  _streamCamera(InputImageRotation rotation, CameraDescription camera) async {
    await _camera.startImageStream((CameraImage image) async {
      if (isDetecting) return;
      isDetecting = true;
      debugPrint('Received camera image with format: ${image.format.group}');
      String res;
      dynamic finalResult = Multimap<String, Face>();

      final inputImage = _inputImageFromCameraImage(image);
      if (inputImage == null) {
        isDetecting = false;
        return;
      }
      final List<Face> faces = await _faceDetector.processImage(inputImage);
      debugPrint('faces ===>>>> ${faces.length}');
      if (faces.isEmpty) {
        faceFound = false;
      } else {
        faceFound = true;
      }

      Face face;
      imglib.Image convertedImage = _convertCameraImage(image, direction);
      for (face in faces) {
        double x, y, w, h;
        x = (face.boundingBox.left - 10);
        y = (face.boundingBox.top - 10);
        w = (face.boundingBox.width + 10);
        h = (face.boundingBox.height + 10);
        imglib.Image croppedImage = imglib.copyCrop(convertedImage, x: x.round(), y: y.round(), width: w.round(), height: h.round());
        croppedImage = imglib.copyResizeCropSquare(croppedImage, size: 112);
        // int startTime = new DateTime.now().millisecondsSinceEpoch;
        res = _recog(croppedImage);
        // int endTime = new DateTime.now().millisecondsSinceEpoch;
        // print("Inference took ${endTime - startTime}ms");
        finalResult.add(res, face);
      }
      debugPrint("Final result: $finalResult");
      setState(() {
        _scanResults = finalResult;
      });
      isDetecting = false;

      // detect(image, _getDetectionMethod(), rotation).then((result) {
      //   debugPrint("Result: $result");
      //   if (result.length == 0) {
      //     faceFound = false;
      //   } else {
      //     faceFound = true;
      //   }

      // }).catchError((_) {
      //     isDetecting = false;
      //   },
      // );
    });
  }

  @override
  void dispose() {
    if (_isCameraInitialized) {
      _faceDetector.close();
      _camera.dispose();
    }
    _name.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Detection'),
        actions: <Widget>[
          PopupMenuButton<Choice>(
            onSelected: (Choice result) {
              if (result == Choice.delete) {
                _resetFile();
              } else {
                _viewLabels();
              }
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<Choice>>[
              const PopupMenuItem<Choice>(
                  value: Choice.view, child: Text('View Saved Faces')),
              const PopupMenuItem<Choice>(
                  value: Choice.delete, child: Text('Remove all faces'))
            ],
          ),
        ],
      ),
      body: _buildImage(),
      floatingActionButton:
          Column(mainAxisAlignment: MainAxisAlignment.end, children: [
        FloatingActionButton(
          backgroundColor: (faceFound) ? Colors.blue : Colors.blueGrey,
          onPressed: () {
            debugPrint("Face found: $faceFound");
            if (faceFound) {
              _addLabel();
            }
          },
          heroTag: null,
          child: const Icon(Icons.add),
        ),
        const SizedBox(height: 10),
        FloatingActionButton(
          onPressed: null,
          heroTag: null,
          child: direction == CameraLensDirection.back
              ? const Icon(Icons.camera_front)
              : const Icon(Icons.camera_rear),
        ),
      ]),
    );
  }

  Widget _buildResults() {
    const Text noResultsText = Text('');
    if (_scanResults == null ||
        !_isCameraInitialized ||
        !_camera.value.isInitialized) {
      return noResultsText;
    }
    CustomPainter painter;

    final Size imageSize = Size(
      _camera.value.previewSize?.height ?? 0,
      _camera.value.previewSize?.width ?? 0,
    );
    painter = FaceDetectorPainter(imageSize, _scanResults);
    return CustomPaint(painter: painter);
  }

  Widget _buildImage() {
    if (!_isCameraInitialized || !_camera.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }
    return Container(
      constraints: const BoxConstraints.expand(),
      child: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          CameraPreview(_camera),
          _buildResults(),
        ],
      ),
    );
  }


  // imglib.Image _convertCameraImage2(CameraImage image, CameraLensDirection dir) {
  //   int width = image.width;
  //   int height = image.height;
  //   // imglib -> Image package from https://pub.dartlang.org/packages/image
  //   var img = imglib.Image(width: width, height: height); // Create Image buffer
  //   // const int hexFF = 0xFF000000;
  //   final int uvyButtonStride = image.planes[1].bytesPerRow;
  //   final int uvPixelStride = image.planes[1].bytesPerPixel ?? 0;
  //   for (int x = 0; x < width; x++) {
  //     for (int y = 0; y < height; y++) {
  //       final int uvIndex =
  //           uvPixelStride * (x / 2).floor() + uvyButtonStride * (y / 2).floor();
  //       final int index = y * width + x;
  //       final yp = image.planes[0].bytes[index];
  //       final up = image.planes[1].bytes[uvIndex];
  //       final vp = image.planes[2].bytes[uvIndex];
  //       // Calculate pixel color
  //       int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
  //       int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
  //           .round()
  //           .clamp(0, 255);
  //       int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
  //       // color: 0x FF  FF  FF  FF
  //       //           A   B   G   R
  //       img.setPixel(x, y, imglib.ColorInt32.rgba(r, g, b, 255));
  //     }
  //   }
  //   var img1 = (dir == CameraLensDirection.front) ? imglib.copyRotate(img, angle: -90) : imglib.copyRotate(img, angle: 90);
  //   return img1;
  // }

  imglib.Image _convertCameraImage(CameraImage image, CameraLensDirection dir) {
    try {
      final int width = image.width;
      final int height = image.height;
      
      if (image.planes.length != 1) {
        debugPrint('Expected single plane for NV21, got ${image.planes.length}');
      }

      // Get the Y plane (first plane)
      final yPlane = image.planes[0];
      final yBuffer = yPlane.bytes;
      
      // Create output image
      var img = imglib.Image(width: width, height: height);

      // Convert YUV to RGB
      int pixelIndex = 0;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          // Get Y value
          final yValue = yBuffer[pixelIndex];
          
          // For simplicity, convert to grayscale first
          // You can expand this to do proper YUV->RGB conversion if needed
          final gray = yValue.clamp(0, 255);
          
          img.setPixel(x, y, imglib.ColorInt32.rgba(gray, gray, gray, 255));
          pixelIndex++;
        }
      }

      // Rotate based on camera direction
      return (dir == CameraLensDirection.front)
          ? imglib.copyRotate(img, angle: -90)
          : imglib.copyRotate(img, angle: 90);

    } catch (e) {
      debugPrint('Error converting camera image: $e');
      // Return a small blank image instead of crashing
      return imglib.Image(width: 112, height: 112);
    }
  }

  String _recog(imglib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, 0.0).reshape([1, 192]);
    interpreter.run(input, output);
    output = output.reshape([192]);
    e1 = List.from(output);
    return compare(e1).toUpperCase();
  }

  String compare(List currEmb) {
    if (data.length == 0) return "No Face saved";
    double minDist = 999;
    double currDist = 0.0;
    String predRes = "NOT RECOGNIZED";
    for (String label in data.keys) {
      currDist = euclideanDistance(data[label], currEmb);
      if (currDist <= threshold && currDist < minDist) {
        minDist = currDist;
        predRes = label;
      }
    }
    debugPrint("$minDist $predRes");
    return predRes;
  }

  void _resetFile() {
    data = {};
    jsonFile.deleteSync();
  }

  void _viewLabels() {
    // setState(() {
    //   _camera = null;
    // });
    String name;
    var alert = AlertDialog(
      title: const Text("Saved Faces"),
      content: ListView.builder(
          padding: const EdgeInsets.all(2),
          itemCount: data.length,
          itemBuilder: (BuildContext context, int index) {
            name = data.keys.elementAt(index);
            return Column(
              children: <Widget>[
                ListTile(
                  title: Text(
                    name,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey[400],
                    ),
                  ),
                ),
                const Padding(
                  padding: EdgeInsets.all(2),
                ),
                const Divider(),
              ],
            );
          }),
      actions: <Widget>[
        TextButton(
          child: const Text("OK"),
          onPressed: () {
            _initializeCamera();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

  void _addLabel() {
    // setState(() {
    //   _camera = null;
    // });
    debugPrint("Adding new face");
    var alert = AlertDialog(
      title: const Text("Add Face"),
      content: Row(
        children: <Widget>[
          Expanded(
            child: TextField(
              controller: _name,
              autofocus: true,
              decoration: const InputDecoration(
                  labelText: "Name", icon: Icon(Icons.face)),
            ),
          )
        ],
      ),
      actions: <Widget>[
        TextButton(
            child: const Text("Save"),
            onPressed: () {
              _handle(_name.text.toUpperCase());
              _name.clear();
              Navigator.pop(context);
            }),
        TextButton(
          child: const Text("Cancel"),
          onPressed: () {
            _initializeCamera();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

  void _handle(String text) {
    data[text] = e1;
    jsonFile.writeAsStringSync(json.encode(data));
    _initializeCamera();
  }
}
