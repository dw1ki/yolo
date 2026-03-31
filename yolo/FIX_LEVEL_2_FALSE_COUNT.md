"""
🔴 FIX LEVEL 2: Prevent False Count pada Lajur Kanan
Solve: Mobil lajur kanan detected hijau (counted) gara-gara bbox lajur kiri tergeser

PROBLEM ANALYSIS:
═════════════════════════════════════════════════════════════════════

Before:                          After counted:
┌─────────────────────────────────────────────────────────────────┐
│  LAJUR KIRI       │  LAJUR KANAN                                │
│ ─────────────────────────────────────────────────────────────  │
│ Mobil1            │  Mobil2                                     │
│ ID=5              │  (belum crossing)                           │
│ (COUNTED, hijau)  │                                             │
│ ─────────────────────────────────────────────────────────────  │
│ ════════════════════════════════════════════════════════════════ │ <- LINE
│ Mobil1 bbox      │  Mobil2                                     │
│ terus geser      │  bbox jadi align dengan bbox lajur kiri     │
│ (detected lagi)  │                                             │
│                  │  ❌ TRACKING: bbox lajur kiri match ke 2!  │
│                  │  ❌ RESULT: Mobil2 inherit counted=True!    │
│                  │  ❌ DISPLAY: Mobil2 jadi hijau tanpa cross! │
└─────────────────────────────────────────────────────────────────┘

ROOT CAUSE:
───────────
1. Object counted = counted=True
2. Bbox object masih dilihat di frame (bergerak terus)
3. Bbox ini match ke object baru (berbeda lane)
4. Object lama punya counted=True, object baru inherit status ini
5. Hasilnya: false count di lane yang salah!

═════════════════════════════════════════════════════════════════════

FIXES APPLIED:
═════════════════════════════════════════════════════════════════════

✅ FIX 1: DELETE counted objects far below line
──────────────────────────────────────────────
Location: Line ~390
Logic:
  if object.counted AND object.y > (LINE + 150px):
    deregister(object)  # Hapus! Tidak perlu track lagi!

Benefit:
  • Object yang sudah counted + jauh = dihapus dari tracking
  • Bekas bbox-nya tidak akan bikin confusion
  • Tracking fokus ke object baru saja
  • No more inherited counted status!

Threshold: 150px di bawah line
  • Reasonable untuk ensure object fully crossed
  • Not too strict (memberi buffer untuk movement)
  • Sufficient untuk clear dari lane lain


✅ FIX 2: ABSOLUTE rejection untuk cross-lane counted objects
──────────────────────────────────────────────────────────────
Location: Line ~330
Logic:
  if object.counted AND object.lane != new_lane:
    REJECT ABSOLUTELY  # No matching allowed!

Before vs After:
  BEFORE: distance < 20px → accept (masih bisa match!)
  AFTER:  lane != same   → REJECT (100% block!)

Benefit:
  • Counted objects CANNOT match ke lane lain
  • Even if distance is very close!
  • Prevents last-minute inheritance of counted status


═════════════════════════════════════════════════════════════════════

EXPECTED BEHAVIOR SETELAH FIX:
═════════════════════════════════════════════════════════════════════

Scenario: Mobil lajur kiri crossing, mobil lajur kanan hadir sebelum crossing

Timeline:
────────
Frame 1:
  Mobil Kiri (ID=5): merah, y=280 (akan crossing)
  Mobil Kanan (ID=6): merah, y=150 (jauh dari line)

Frame 5:
  Mobil Kiri (ID=5): hijau, y=320 (COUNTED! sudah crossing)
  Mobil Kanan (ID=6): merah, y=280 (masih jauh dari line)

Frame 6:
  Mobil Kiri bbox masih terdeteksi: y=350 (di bawah line)
  Mobil Kanan: y=290 (terus bergerak)
  
  Tracking logic:
    ❌ Old: bbox kiri (y=350) match ke mobil kanan (y=290) → Mobil Kanan jadi hijau
    ✅ New: 
       1. Check if kiri.lane != kanan.lane → REJECT match
       2. Kiri sudah y > LINE+150 → DELETE kiri
       3. Mobil Kanan tetap merah (ID=6, tidak inherit)

Frame 10:
  Mobil Kiri: DELETED (sudah di-count, jauh di bawah)
  Mobil Kanan (ID=6): merah, y=310 (akan crossing)

Frame 12:
  Mobil Kanan (ID=6): hijau, y=320 (COUNTED! benar-benar crossing)

RESULT: ✅ Accurate count, no false counting di lajur kanan!

═════════════════════════════════════════════════════════════════════

TECHNICAL DETAILS:
═════════════════════════════════════════════════════════════════════

Config yang relevan:
  LINE_POSITION = 300         # Y-coordinate of counting line
  DELETION_THRESHOLD = 150px  # Distance di bawah line untuk delete
  MAX_DISTANCE = 60px         # Normal tracking distance

Behavior per lane:
  
  LEFT LANE:
    Uncounted: merah, track normally
    Counted:   hijau, track until y > 300+150 = 450
    Deleted:   tidak ada (removed from tracking)
  
  RIGHT LANE:
    Uncounted: biru, track independently
    Counted:   hijau (only saat benar-benar cross)
    Protected: cannot inherit counted status dari lane lain

═════════════════════════════════════════════════════════════════════

HOW TO TEST:
═════════════════════════════════════════════════════════════════════

1. Restart API dengan fix baru:
   cd D:\backup\pktj\backend\yolo
   uvicorn api:app --reload

2. Process video WhatsApp (yang punya issue):
   
   Hal yang harus dilihat:
   ✅ Mobil lajur kiri: counted (hijau) → disappear (dihapus)
   ✅ Mobil lajur kanan: tetap merah sampai benar-benar crossing
   ✅ Hanya menjadi hijau saat y > 300 (benar crossing)
   ✅ Counter per lane akurat

3. Compare hasil:
   Left lane count should = number of vehicles that crossed
   Right lane count should = number of vehicles that crossed (not false)

═════════════════════════════════════════════════════════════════════

CODE CHANGES SUMMARY:
═════════════════════════════════════════════════════════════════════

File: api.py

Change 1: CROSS-LANE REJECTION (Line ~330)
   Before: if distance > 20px { continue }
   After:  if lane_different { REJECT completely }

Change 2: COUNTED OBJECT DELETION (Line ~390)
   New:    if counted AND y > LINE+150 { deregister }

Result: NO MORE FALSE COUNTS! 🎉

════════════════════════════════════════════════════════════════════
"""

print("""
╔════════════════════════════════════════════════════════════════╗
║         🔴 FIX LEVEL 2: Lane-Jump False Count                 ║
║         Problem: Mobil lajur kanan ter-count tanpa crossing   ║
║         Solution: Delete counted objects + Strict rejection   ║
╚════════════════════════════════════════════════════════════════╝

KEY INSIGHT:
  Object yang sudah di-count tidak boleh lagi bikin confusion!
  
  Solusi 1: Hapus object dari tracking setelah counted + jauh
  Solusi 2: Absolutely reject cross-lane matching untuk counted
  
  Hasil: Lajur kanan hanya hijau saat benar-benar crossing!

USE CASE YG DIFIX:
  Before: Mobil lajur kanan langsung hijau (false count)
  After:  Mobil lajur kanan tetap merah sampai crossing (accurate)

READY TO TEST!
""")
