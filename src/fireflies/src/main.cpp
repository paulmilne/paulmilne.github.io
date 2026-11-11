// ESP32 fireflies: Mirollo-Strogatz pulse-coupled oscillators over ESP-NOW broadcast.
//
// Each node runs a clock with its own period. When its phase reaches 1 it "flashes":
// LED on, and a broadcast packet to every other node. A node that hears a flash
// kicks its state x = f(phase) up by EPS, where f is concave, so the same kick
// advances the phase more late in the cycle. If the kick reaches the threshold
// the node flashes at once. Flashes heard within REFRACTORY_MS of the node's own
// flash are ignored: without that, radio delay turns the group's own flashes into
// an endless echo (see the post).
//
// Build with -DLOGGER for a listen-only board that prints "t_us,node,seq" for
// every flash it hears, for plotting a raster of the whole group.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_arduino_version.h>
#include <math.h>

#ifndef LED_PIN
#define LED_PIN 2
#endif
#ifdef LED_ACTIVE_LOW            // e.g. ESP32-C3 SuperMini: blue LED on GPIO 8, lit when LOW
#define LED_ON  LOW
#define LED_OFF HIGH
#else
#define LED_ON  HIGH
#define LED_OFF LOW
#endif

static const float    PERIOD_MS      = 1000.0f;  // nominal period
static const float    DETUNE         = 0.02f;    // each node's period is off by up to +-2 %
static const float    EPS            = 0.05f;    // state kick per flash heard
static const float    B              = 3.0f;     // curvature of f
static const uint32_t REFRACTORY_MS  = 50;       // deaf time after own flash
static const uint32_t LED_ON_MS      = 60;
static const uint8_t  WIFI_CHANNEL   = 1;
static const uint8_t  BROADCAST[6]   = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

struct __attribute__((packed)) Flash {
  uint32_t magic;   // 'FFLY'
  uint8_t  node;
  uint16_t seq;
};
static const uint32_t MAGIC = 0x46464C59;

// ---- received flashes: written in the Wi-Fi task, read in loop() ----
static volatile uint32_t rxCount = 0;
static portMUX_TYPE rxMux = portMUX_INITIALIZER_UNLOCKED;
#ifdef LOGGER
struct Heard { uint32_t t_us; uint8_t node; uint16_t seq; };
static Heard ring[64];
static volatile uint8_t head = 0, tail = 0;
#endif

static void onFlash(const uint8_t *data, int len) {
  if (len != sizeof(Flash)) return;
  Flash f; memcpy(&f, data, sizeof f);
  if (f.magic != MAGIC) return;
  portENTER_CRITICAL(&rxMux);
  rxCount++;
#ifdef LOGGER
  ring[head] = {(uint32_t)micros(), f.node, f.seq};
  head = (head + 1) & 63;
#endif
  portEXIT_CRITICAL(&rxMux);
}

#if ESP_ARDUINO_VERSION_MAJOR >= 3
static void onRecv(const esp_now_recv_info_t *, const uint8_t *data, int len) { onFlash(data, len); }
#else
static void onRecv(const uint8_t *, const uint8_t *data, int len) { onFlash(data, len); }
#endif

// ---- oscillator ----
static float    periodMs;
static uint32_t cycleStartMs, lastFlashMs = 0, seenCount = 0;
static uint16_t seq = 0;
static uint8_t  nodeId;

static float f(float p)    { return logf(1.0f + (expf(B) - 1.0f) * p) / B; }
static float finv(float x) { return (expf(B * x) - 1.0f) / (expf(B) - 1.0f); }

static void flash(uint32_t now) {
  digitalWrite(LED_PIN, LED_ON);
  Flash pkt{MAGIC, nodeId, seq++};
  esp_now_send(BROADCAST, (const uint8_t *)&pkt, sizeof pkt);
  cycleStartMs = now;
  lastFlashMs = now;
  Serial.printf("F,%u,%u,%lu\n", nodeId, pkt.seq, (unsigned long)now);   // for tools/fireflies_log.py
}

void setup() {
  Serial.begin(115200);
#if ARDUINO_USB_CDC_ON_BOOT && ARDUINO_USB_MODE
  Serial.setTxTimeoutMs(0);   // never block on USB serial when no computer is reading
#endif
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LED_OFF);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  if (esp_now_init() != ESP_OK) { Serial.println("ESP-NOW init failed"); while (true) delay(1000); }
  esp_now_register_recv_cb(onRecv);
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BROADCAST, 6);
  peer.channel = WIFI_CHANNEL;
  peer.encrypt = false;
  esp_now_add_peer(&peer);

  uint8_t mac[6]; WiFi.macAddress(mac);
  nodeId = mac[5];
  float u = (esp_random() / 4294967295.0f) * 2.0f - 1.0f;   // -1..1
  periodMs = PERIOD_MS * (1.0f + DETUNE * u);
  cycleStartMs = millis() - (uint32_t)(periodMs * (esp_random() / 4294967295.0f));  // random start phase
#ifdef LOGGER
  Serial.println("t_us,node,seq");
#else
  Serial.printf("I,%u,%.1f\n", nodeId, periodMs);                         // node id, period
#endif
}

void loop() {
#ifdef LOGGER
  while (tail != head) {
    Heard h = ring[tail]; tail = (tail + 1) & 63;
    Serial.printf("%lu,%u,%u\n", (unsigned long)h.t_us, h.node, h.seq);
    digitalWrite(LED_PIN, LED_ON); delay(2); digitalWrite(LED_PIN, LED_OFF);
  }
  return;
#else
  uint32_t now = millis();
  if (now - lastFlashMs >= LED_ON_MS) digitalWrite(LED_PIN, LED_OFF);

  uint32_t heard;
  portENTER_CRITICAL(&rxMux); heard = rxCount; portEXIT_CRITICAL(&rxMux);
  uint32_t newFlashes = heard - seenCount;
  seenCount = heard;

  float phase = (now - cycleStartMs) / periodMs;
  if (phase >= 1.0f) { flash(now); return; }

  if (newFlashes && (now - lastFlashMs) >= REFRACTORY_MS) {
    float x = f(phase) + EPS * newFlashes;
    if (x >= 1.0f) { flash(now); return; }                  // absorbed: fire with the group
    float p = finv(x);
    cycleStartMs = now - (uint32_t)(p * periodMs);           // jump the phase forward
  }
#endif
}
