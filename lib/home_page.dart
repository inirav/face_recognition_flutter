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

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
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

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.accurate,
    ),
  );

  @override
  void initState() {
    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    _initializeCamera();
    super.initState();
  }

  Future<void> _loadModel() async {
    try {
      final gpuDelegateV2 = GpuDelegateV2(options: GpuDelegateOptionsV2());
      var interpreterOptions = InterpreterOptions()..addDelegate(gpuDelegateV2);
      interpreter = await Interpreter.fromAsset('assets/mobilefacenet.tflite',
          options: interpreterOptions);
    } on Exception {
      debugPrint('Error loading model');
    }
  }

  Future<CameraDescription> _getCameraDescription() async {
    List<CameraDescription> cameras = await availableCameras();
    return cameras.firstWhere((CameraDescription camera) =>
        camera.lensDirection == CameraLensDirection.front);
  }

  _initializeCamera() async {
    await _loadModel();
    CameraDescription camera = await _getCameraDescription();
    _camera =
        CameraController(camera, ResolutionPreset.high, enableAudio: false);
    InputImageRotation rotation =
        rotationIntToImageRotation(camera.sensorOrientation);

    try {
      await _camera.initialize();
      setState(() {
        _isCameraInitialized = true;
      });
      await Future.delayed(const Duration(milliseconds: 500));
      tempDir = await getApplicationDocumentsDirectory();
      String embPath = '${tempDir!.path}/emb.json';
      jsonFile = File(embPath);
      if (jsonFile.existsSync())
        data = json.decode(jsonFile.readAsStringSync());
      _streamCamera(rotation, camera);
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<List<Face>> detectFacesFromImage(
      CameraImage image, CameraDescription camera) async {
    if (image.planes.isEmpty) {
      throw ArgumentError('Invalid image data');
    }

    InputImageMetadata firebaseImageMetadata = InputImageMetadata(
      rotation: rotationIntToImageRotation(camera.sensorOrientation),
      format: InputImageFormatValue.fromRawValue(image.format.raw) ??
          InputImageFormat.yuv420,
      size: Size(image.width.toDouble(), image.height.toDouble()),
      bytesPerRow: image.planes[0].bytesPerRow,
    );
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    InputImage firebaseVisionImage =
        InputImage.fromBytes(bytes: bytes, metadata: firebaseImageMetadata);
    List<Face> faces = await _faceDetector.processImage(firebaseVisionImage);
    debugPrint('faces ${faces.length}');
    return faces;
  }

  _streamCamera(InputImageRotation rotation, CameraDescription camera) async {
    await _camera.startImageStream((CameraImage image) async {
      if (isDetecting) return;
      isDetecting = true;
      debugPrint('Received camera image with format: ${image.format.group}');
      String res;
      dynamic finalResult = Multimap<String, Face>();

      detect(image, _getDetectionMethod(), rotation).then((result) {
        debugPrint("Result: $result");
        if (result.length == 0) {
          faceFound = false;
        } else {
          faceFound = true;
        }
        // Face face;
        // imglib.Image convertedImage = _convertCameraImage(image, direction);
        // for (face in result) {
        //   double x, y, w, h;
        //   x = (face.boundingBox.left - 10);
        //   y = (face.boundingBox.top - 10);
        //   w = (face.boundingBox.width + 10);
        //   h = (face.boundingBox.height + 10);
        //   imglib.Image croppedImage = imglib.copyCrop(convertedImage, x: x.round(), y: y.round(), width: w.round(), height: h.round());
        //   croppedImage = imglib.copyResizeCropSquare(croppedImage, size: 112);
        //   // int startTime = new DateTime.now().millisecondsSinceEpoch;
        //   res = _recog(croppedImage);
        //   // int endTime = new DateTime.now().millisecondsSinceEpoch;
        //   // print("Inference took ${endTime - startTime}ms");
        //   finalResult.add(res, face);
        // }
        // debugPrint("Final result: $finalResult");
        // setState(() {
        //   _scanResults = finalResult;
        // });
        // isDetecting = false;
      }).catchError(
        (_) {
          isDetecting = false;
        },
      );
    });
  }

  @override
  void dispose() {
    if (_isCameraInitialized) {
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

  // void _toggleCameraDirection() async {
  //   if (direction == CameraLensDirection.back) {
  //     direction = CameraLensDirection.front;
  //   } else {
  //     direction = CameraLensDirection.back;
  //   }
  //   await _camera.stopImageStream();
  //   await _camera.dispose();
  //   setState(() {
  //     _camera = null;
  //   });
  //   _initializeCamera();
  // }

  imglib.Image _convertCameraImage(CameraImage image, CameraLensDirection dir) {
    try {
      final int width = image.width;
      final int height = image.height;

      // Log image dimensions and format
      debugPrint('Image dimensions: ${width}x$height');
      debugPrint('Image format: ${image.format.group}');
      debugPrint('Number of planes: ${image.planes.length}');

      // Validate image format and planes
      if (image.format.group != ImageFormatGroup.yuv420) {
        throw Exception('Unsupported image format: ${image.format.group}');
      }

      if (image.planes.length != 3) {
        throw Exception('Invalid number of planes: ${image.planes.length}');
      }

      // Validate plane data
      for (int i = 0; i < image.planes.length; i++) {
        if (image.planes[i].bytes.isEmpty) {
          throw Exception('Plane $i is empty');
        }
        debugPrint('Plane $i size: ${image.planes[i].bytes.length}');
      }

      // Rest of the conversion code remains the same...
      final plane0 = image.planes[0].bytes;
      final plane1 = image.planes[1].bytes;
      final plane2 = image.planes[2].bytes;

      final int pixelCount = width * height;
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

      var rgbBytes = List<int>.filled(pixelCount * 3, 0);
      int rgbIndex = 0;

      // Add safety check for buffer sizes
      if (plane0.length < pixelCount ||
          plane1.length < (pixelCount / 4).ceil()) {
        throw Exception('Insufficient buffer size for image conversion');
      }

      for (int row = 0; row < height; row++) {
        int uvRow = (row / 2).floor();

        for (int col = 0; col < width; col++) {
          final int uvCol = (col / 2).floor();

          // Index calculations
          final yIndex = row * width + col;
          final uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride;

          // Skip if out of bounds
          if (yIndex >= plane0.length || uvIndex >= plane1.length) continue;

          // YUV values
          final int y = plane0[yIndex];
          final int u = plane1[uvIndex];
          final int v = plane2[uvIndex];

          // Basic YUV to RGB conversion
          final int r = ((y + (1.370705 * (v - 128)))).round().clamp(0, 255);
          final int g = ((y - (0.698001 * (v - 128)) - (0.337633 * (u - 128))))
              .round()
              .clamp(0, 255);
          final int b = ((y + (1.732446 * (u - 128)))).round().clamp(0, 255);

          rgbBytes[rgbIndex++] = r;
          rgbBytes[rgbIndex++] = g;
          rgbBytes[rgbIndex++] = b;
        }
      }

      // Create image from RGB bytes
      var img = imglib.Image.fromBytes(
        width: width,
        height: height,
        bytes: Uint8List.fromList(rgbBytes).buffer,
        order: imglib.ChannelOrder.rgb,
      );

      // Handle rotation based on camera direction
      var rotatedImage = (dir == CameraLensDirection.front)
          ? imglib.copyRotate(img, angle: -90)
          : imglib.copyRotate(img, angle: 90);

      return rotatedImage;
    } catch (e, stackTrace) {
      // Enhanced error logging
      debugPrint('Error converting camera image: $e');
      debugPrint('Stack trace: $stackTrace');
      // Return a small blank image instead of crashing
      return imglib.Image(width: 112, height: 112);
    }
  }

  String _recog(imglib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, 0.0);
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

  HandleDetection _getDetectionMethod() {
    final faceDetector = FaceDetector(
        options: FaceDetectorOptions(
            enableLandmarks: true,
            performanceMode: FaceDetectorMode.accurate,
            enableTracking: true));
    // return faceDetector.processImage;
    return (InputImage inputImage) {
      // Ensure proper cleanup after processing
      try {
        return faceDetector.processImage(inputImage);
      } catch (e) {
        debugPrint('Face detection error: $e');
        return Future.value(<Face>[]);
      } finally {
        // Optional: Close detector when done if needed
        // faceDetector.close();
      }
    };
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
