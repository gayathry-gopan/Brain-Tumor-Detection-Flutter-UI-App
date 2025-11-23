import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';

void main() {
  runApp(const BrainTumorUIApp());
}

class BrainTumorUIApp extends StatelessWidget {
  const BrainTumorUIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: "Brain Tumor Detector",
      debugShowCheckedModeBanner: false,
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Interpreter? interpreter;
  List<String> labels = [];
  File? imageFile;
  String result = "";
  double confidence = 0.0;
  bool loading = false;

  final picker = ImagePicker();
  final int inputSize = 224;

  @override
  void initState() {
    super.initState();
    loadModel();
    loadLabels();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(
        "assets/final_mobilenet_brain_tumor.tflite",
      );
      print("Model loaded");
    } catch (e) {
      print("Model load error: $e");
    }
  }

  Future<void> loadLabels() async {
    final raw = await rootBundle.loadString("assets/labels.txt");
    labels = raw.split("\n").where((e) => e.trim().isNotEmpty).toList();
  }

  Future<void> pickImage() async {
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      imageFile = File(picked.path);
      result = "";
      confidence = 0;
    });

    runInference(imageFile!);
  }

  Future<void> runInference(File file) async {
    if (interpreter == null) return;

    setState(() => loading = true);

    Uint8List raw = await file.readAsBytes();
    img.Image? decoded = img.decodeImage(raw);

    img.Image resized =
        img.copyResize(decoded!, width: inputSize, height: inputSize);

    var input = List.generate(
      1,
      (_) => List.generate(
          inputSize,
          (_) => List.generate(
                inputSize,
                (_) => List.filled(3, 0.0),
              )),
    );

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final px = resized.getPixel(x, y);
        input[0][y][x][0] = px.r / 255;
        input[0][y][x][1] = px.g / 255;
        input[0][y][x][2] = px.b / 255;
      }
    }

    var output = List.generate(1, (_) => List.filled(labels.length, 0.0));
    interpreter!.run(input, output);

    List<double> probs = output[0];
    int maxIdx = probs.indexOf(probs.reduce((a, b) => a > b ? a : b));

    setState(() {
      result = labels[maxIdx];
      confidence = probs[maxIdx] * 100;
      loading = false;
    });
  }

  // ---------- UI STARTS HERE ----------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      backgroundColor: const Color(0xFF0D0F21),
      body: Stack(
        children: [
          // Background Gradient
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Color(0xFF16222A), Color(0xFF3A6073)],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
          ),

          // MAIN CONTENT
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    const Text(
                      "Brain Tumor Detector",
                      style: TextStyle(
                        fontSize: 32,
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 25),

                    // Glassmorphism Image Box
                    Container(
                      height: 280,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(25),
                        color: Colors.white.withOpacity(0.08),
                        border: Border.all(color: Colors.white24, width: 1),
                        boxShadow: const [
                          BoxShadow(
                            color: Colors.black26,
                            blurRadius: 10,
                            offset: Offset(2, 4),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(25),
                        child: imageFile == null
                            ? const Center(
                                child: Text(
                                  "No Image Selected",
                                  style: TextStyle(
                                    fontSize: 18,
                                    color: Colors.white70,
                                  ),
                                ),
                              )
                            : Image.file(
                                imageFile!,
                                fit: BoxFit.cover,
                              ),
                      ),
                    ),

                    const SizedBox(height: 30),

                    // Select Button
                    ElevatedButton(
                      onPressed: loading ? null : pickImage,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF00C6FF),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 40, vertical: 14),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
                      ),
                      child: Text(
                        loading ? "Processing..." : "Select MRI Image",
                        style: const TextStyle(
                            color: Colors.black,
                            fontSize: 18,
                            fontWeight: FontWeight.bold),
                      ),
                    ),

                    const SizedBox(height: 30),

                    // Loader
                    if (loading)
                      const CircularProgressIndicator(
                        strokeWidth: 4,
                        color: Colors.white,
                      ),

                    // Result Card
                    if (!loading && result.isNotEmpty)
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        padding: const EdgeInsets.all(20),
                        margin: const EdgeInsets.only(top: 20),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(18),
                          color: Colors.white.withOpacity(0.15),
                          border: Border.all(color: Colors.white30),
                        ),
                        child: Column(
                          children: [
                            const Text(
                              "Prediction Result",
                              style: TextStyle(
                                fontSize: 22,
                                color: Colors.white70,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 10),
                            Text(
                              result,
                              style: const TextStyle(
                                fontSize: 26,
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 10),
                            Text(
                              "Confidence Score: ${confidence.toStringAsFixed(2)}%",
                              style: const TextStyle(
                                fontSize: 18,
                                color: Colors.greenAccent,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
