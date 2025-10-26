



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
  
  void ejecutarPaso(float t) {
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