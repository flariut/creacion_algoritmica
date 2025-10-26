// ==========================
//  Secuenciador de imágenes
//  tipo Elektron (simplificado)
// ==========================

Secuenciador seq;
String modoTest = "simple"; 
// opciones: "simple", "fade", "prob", "random"

void setup() {
  size(800, 800); 
  frameRate(60);
  seq = new Secuenciador("imagenes", 4, 60); // carpeta, pasos, bpm
  
  if (modoTest.equals("simple")) {
    seq.aleatorizarPasos();
  } else {
    seq.testMode(modoTest);
  }
}

void draw() {
  background(0);
  seq.run(2);  // ejecutar con 8 repeticiones
}

// ==========================
// Clase Paso
// ==========================
class Paso {
  boolean encendido;
  float probabilidad; // 0 a 1
  float fadeIn, fadeOut; // porcentaje entre 0 y 1
  float hueShift;
  int subdivisiones;
  PImage img;
  
  Paso(PImage img_) {
    img = img_;
    encendido = true;
    probabilidad = 1.0;
    fadeIn = 0.0;
    fadeOut = 0.0;
    hueShift = 0.0;
    subdivisiones = 1;
  }
  
  void mostrarImagen(float t) {
    if (!encendido) return;
    if (random(1) > probabilidad) return;
    
    // Calcular alpha con fade in/out
    float alpha = 255;
    if (t < fadeIn) {
      if (fadeIn > 0) {
        alpha = map(t, 0, fadeIn, 0, 255);
      } else {
        alpha = 255;
      }
    } else if (t > 1.0 - fadeOut) {
      if (fadeOut > 0) {
        alpha = map(t, 1.0 - fadeOut, 1.0, 255, 0);
      } else {
        alpha = 0;
      }
    }
    
    tint(255, alpha);
    
    // Escalar para llenar y recortar
    float scale = max(width / float(img.width), height / float(img.height));
    int w = int(img.width * scale);
    int h = int(img.height * scale);
    
    imageMode(CENTER);
    image(img, width/2, height/2, w, h);
    imageMode(CORNER);
    
    noTint();
  }
}

// ==========================
// Clase Secuenciador
// ==========================
class Secuenciador {
  int numPasos;
  float bpm;
  String carpeta;
  Paso[] pasos;
  ArrayList<PImage> imagenes;
  int currentStep = 0;
  int repeticiones = 0;
  float stepDuration; // en segundos
  float stepStartTime;
  
  Secuenciador(String carpeta_, int numPasos_, float bpm_) {
    carpeta = carpeta_;
    numPasos = numPasos_;
    bpm = bpm_;
    stepDuration = 60.0 / bpm;
    cargarImagenes();
    pasos = new Paso[numPasos];
    for (int i=0; i<numPasos; i++) {
      PImage img = imagenes.get(int(random(imagenes.size())));
      pasos[i] = new Paso(img);
    }
    stepStartTime = millis()/1000.0;
  }
  
  void cargarImagenes() {
    imagenes = new ArrayList<PImage>();
  
    File folder = new File(dataPath(carpeta));
    if (!folder.exists()) {
      println("La carpeta no existe: " + folder.getAbsolutePath());
      return;
    }
  
    File[] files = folder.listFiles();
    if (files == null) {
      println("No se pudo leer la carpeta: " + folder.getAbsolutePath());
      return;
    }
  
    println("Archivos encontrados en " + folder.getAbsolutePath() + ":");
    for (File f : files) {
      println(" - " + f.getName());
      if (f.getName().toLowerCase().endsWith(".jpg") || f.getName().toLowerCase().endsWith(".png")) {
        PImage img = loadImage(f.getAbsolutePath());
        if (img != null) {
          imagenes.add(img);
          println("cargada");
        } else {
          println("error al cargar");
        }
      }
    }

    if (imagenes.size() == 0) {
      println("Ninguna imagen cargada.");
    }
  }

  void aleatorizarPasos() {
    for (int i=0; i<numPasos; i++) {
      PImage img = imagenes.get(int(random(imagenes.size())));
      pasos[i] = new Paso(img);
      pasos[i].encendido = random(1) > 0.3;
      pasos[i].probabilidad = random(1);
      pasos[i].fadeIn = random(0.1, 0.5);
      pasos[i].fadeOut = random(0.1, 0.5);
      pasos[i].hueShift = random(360);
      pasos[i].subdivisiones = int(random(1, 4));
    }
  }
  
  void testMode(String modo) {
    for (int i=0; i<numPasos; i++) {
      PImage img = imagenes.get(int(random(imagenes.size())));
      pasos[i] = new Paso(img);
      
      if (modo.equals("simple")) {
        pasos[i].encendido = true;
        pasos[i].probabilidad = 1.0;
      }
      else if (modo.equals("fade")) {
        pasos[i].encendido = true;
        pasos[i].fadeIn = 0.3;
        pasos[i].fadeOut = 0.3;
      }
      else if (modo.equals("prob")) {
        pasos[i].encendido = true;
        pasos[i].probabilidad = 0.5;
      }
    }
  }
  
  void run(int reps) {
    float now = millis()/1000.0;
    float t = (now - stepStartTime) / stepDuration; // progreso 0..1
    
    if (t > 1.0) {
      currentStep++;
      stepStartTime = now;
      if (currentStep >= numPasos) {
        currentStep = 0;
        repeticiones++;
        if (repeticiones >= reps) {
          repeticiones = 0;
          aleatorizarPasos(); // nueva secuencia
        }
      }
    }
    
    // Mostrar paso actual
    pasos[currentStep].mostrarImagen(t);
  }
}
