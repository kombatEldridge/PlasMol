/**
 * PlasMol dissertation campaign — jobs/final
 * Run: node presentation.js
 */

const fs = require("fs");
const pptxgen = require("pptxgenjs");
const FIG = "/Users/bldrdge1/Downloads/repos/PlasMol/jobs/final/figures";

const COLOR = {
  darkBg: "1B2420",
  cream: "F1EDE3",
  beige: "E8DFC8",
  cardCream: "F5EFE2",
  ink: "1B2420",
  body: "3B3F3A",
  muted: "7A7F76",
  mutedLight: "A7A79A",
  forest: "3B5D3A",
  moss: "7A9A6A",
  terracotta: "C26A33",
  hairline: "C9C4B4",
  hairlineDark: "2A3530",
  subtitleOnDark: "C9C4B4",
  titleOnDark: "EFE9D9",
  panel: "EBE6D6",
};

const FONT = { head: "Georgia", body: "Arial", mono: "Menlo" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Nanoparticle and CAP effects on a double core hole";
pres.author = "PlasMol";
pres.subject = "jobs/final campaign plan";

const W = 13.333;
const H = 7.5;
const MARGIN_X = 0.7;
const N = "13";

function addEyebrow(slide, label, page, onDark = false) {
  const labelColor = onDark ? COLOR.mutedLight : COLOR.muted;
  slide.addText(label, {
    x: MARGIN_X, y: 0.32, w: 9.2, h: 0.28,
    fontFace: FONT.body, fontSize: 11, color: labelColor,
    charSpacing: 2.2, margin: 0,
  });
  slide.addText(page, {
    x: W - MARGIN_X - 1.6, y: 0.32, w: 1.6, h: 0.28,
    fontFace: FONT.body, fontSize: 11, color: labelColor,
    align: "right", margin: 0,
  });
}

function addTitle(slide, text, y, h, size) {
  slide.addText(text, {
    x: MARGIN_X, y, w: W - 2 * MARGIN_X, h,
    fontFace: FONT.head, fontSize: size, color: COLOR.ink,
    margin: 0, valign: "top",
  });
}

// ----------------------------------------------------------------------
// 1  Cover
// ----------------------------------------------------------------------
function slideCover() {
  const s = pres.addSlide();
  s.background = { color: COLOR.darkBg };
  addEyebrow(s, "PLASMOL   ·   JOBS / FINAL", "01  /  " + N, true);

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X, y: 1.15, w: 3.35, h: 0.38,
    fill: { color: COLOR.darkBg },
    line: { color: COLOR.mutedLight, width: 0.75 },
  });
  s.addText("DISSERTATION CAMPAIGN", {
    x: MARGIN_X, y: 1.15, w: 3.35, h: 0.38,
    fontFace: FONT.body, fontSize: 11, color: COLOR.subtitleOnDark,
    align: "center", valign: "middle", margin: 0, charSpacing: 1.4,
  });

  s.addText("Nanoparticle and CAP effects\non a double core hole", {
    x: MARGIN_X, y: 1.75, w: 11.2, h: 2.15,
    fontFace: FONT.head, fontSize: 40, color: COLOR.titleOnDark, margin: 0,
  });

  s.addText("A sudden double vacancy on one core orbital, next to a nanoparticle. The molecule and the sphere are not chosen. Each later run turns one piece on and leaves the rest fixed.", {
    x: MARGIN_X, y: 4.15, w: 10.6, h: 0.85,
    fontFace: FONT.body, fontSize: 16, color: COLOR.subtitleOnDark, margin: 0,
  });

  s.addShape(pres.shapes.LINE, {
    x: MARGIN_X, y: 5.85, w: W - 2 * MARGIN_X, h: 0,
    line: { color: COLOR.hairlineDark, width: 0.75 },
  });

  const meta = [
    ["SERIES", "Sudden single-site DCH"],
    ["PAIR", "Not chosen  ·  two pairs still open"],
    ["STATUS", "Step 1 done  ·  2–8 are placeholders"],
  ];
  meta.forEach((pair, i) => {
    const x = MARGIN_X + i * 4.05;
    s.addText(pair[0], {
      x, y: 6.05, w: 3.8, h: 0.24,
      fontFace: FONT.body, fontSize: 11, color: COLOR.mutedLight, margin: 0, charSpacing: 1.5,
    });
    s.addText(pair[1], {
      x, y: 6.32, w: 3.9, h: 0.32,
      fontFace: FONT.body, fontSize: 14, bold: true, color: COLOR.titleOnDark, margin: 0,
    });
  });

  s.addNotes("Open with the comparison, not a chosen dye. The molecule and the nanoparticle are unset. Step 1, the literature and the valence screen, is done. Steps 2–7 have inputs whose molecule and nanoparticle fields are still placeholders. Do not launch a file that contains PLACEHOLDER_.");
}

// ----------------------------------------------------------------------
// 2  The question
// ----------------------------------------------------------------------
function slideQuestion() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "THE QUESTION", "02  /  " + N);
  addTitle(s, "One switch at a time, or the causes mix.", 0.72, 0.55, 30);

  const cards = [
    ["01", "The nanoparticle", "The local field, and, when back-propagation is on, the field the molecule itself radiates into the cell."],
    ["02", "The CAP", "The Lopata non-Hermitian term on the Fock matrix. It sets lifetimes and peak widths, instead of a coherent oscillation that does not decay."],
    ["03", "Then two checks", "Step 7 moves the gap on the resonant pair. Step 8 repeats the key runs where the plasmon misses the bright root."],
  ];
  cards.forEach((c, i) => {
    const x = MARGIN_X + i * 4.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.5, w: 3.88, h: 2.55,
      fill: { color: COLOR.cardCream },
      line: { color: COLOR.hairline, width: 0.75 },
    });
    s.addText(c[0], {
      x: x + 0.22, y: 1.64, w: 3.4, h: 0.28,
      fontFace: FONT.body, fontSize: 12, color: COLOR.terracotta, margin: 0, charSpacing: 1.2,
    });
    s.addText(c[1], {
      x: x + 0.22, y: 1.96, w: 3.45, h: 0.42,
      fontFace: FONT.head, fontSize: 18, color: COLOR.ink, margin: 0,
    });
    s.addText(c[2], {
      x: x + 0.22, y: 2.48, w: 3.45, h: 1.35,
      fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0,
    });
  });

  const cols = [
    ["DYNAMICS", "Hole occupations on the neutral MO basis. The sudden double hole is already non-stationary, so the hole moves with no external field."],
    ["SPECTRA", "A δ-kick on the isolated molecule, and a hybrid cross section under a Gaussian centered on that molecule’s bright root."],
  ];
  cols.forEach((c, i) => {
    const x = MARGIN_X + i * 6.05;
    s.addText(c[0], {
      x, y: 4.28, w: 5.7, h: 0.28,
      fontFace: FONT.body, fontSize: 12, color: COLOR.forest, margin: 0, charSpacing: 1.4,
    });
    s.addText(c[1], {
      x, y: 4.58, w: 5.8, h: 0.85,
      fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X, y: 5.85, w: W - 2 * MARGIN_X, h: 1.15,
    fill: { color: COLOR.darkBg },
  });
  s.addText("NOT A SPECTRUM", {
    x: MARGIN_X + 0.28, y: 5.85, w: 2.3, h: 1.15,
    fontFace: FONT.body, fontSize: 12, color: COLOR.moss, valign: "middle", margin: 0, charSpacing: 1.1,
  });
  s.addText("A field-free hybrid job has no incident field. The cross section divides by that field, so the denominator is ~0. Those runs are occupations, the local field, and the induced dipole.", {
    x: MARGIN_X + 2.7, y: 5.98, w: 9.1, h: 0.9,
    fontFace: FONT.head, fontSize: 15, italic: true, color: COLOR.titleOnDark, margin: 0,
  });

  s.addNotes("The published result is a difference. DCH plus nanoparticle plus CAP plus a pulse mixes four causes. Dynamics are mo_occ.csv. Spectra are cross_section. If there is no incident source, do not divide.");
}

// ----------------------------------------------------------------------
// 3  Already checked
// ----------------------------------------------------------------------
function slideChecked() {
  const s = pres.addSlide();
  s.background = { color: COLOR.beige };
  addEyebrow(s, "ALREADY CHECKED   ·   NOT PART OF JOBS / FINAL", "03  /  " + N);
  addTitle(s, "The methods line up with the references.", 0.62, 0.46, 28);

  const colW = 3.95;
  const gap = 0.18;
  const imgH = 3.05;
  const imgTop = 1.72;
  const panels = [
    {
      k: "01",
      title: "Sudden DCH",
      image: FIG + "/dch_nascimento.png",
      aspect: 1067 / 781,
      cap: "3-pentanone holes. Solid: PlasMol. Dashed: Nascimento, Fig. 8a.",
    },
    {
      k: "02",
      title: "The CAP",
      image: FIG + "/TDDFT_iGamma.png",
      aspect: 3284 / 2055,
      cap: "Water with the CAP, set against Lopata and Govind, JCTC 2013.",
    },
    {
      k: "03",
      title: "Hybrid loop",
      image: FIG + "/gersten_nitzan.png",
      aspect: 1873 / 1411,
      cap: "Sodium beside gold. PlasMol against the Gersten–Nitzan model.",
    },
  ];

  panels.forEach((p, i) => {
    const x = MARGIN_X + i * (colW + gap);
    s.addText(p.k + "   " + p.title.toUpperCase(), {
      x, y: 1.22, w: colW, h: 0.28,
      fontFace: FONT.body, fontSize: 12, color: COLOR.terracotta, margin: 0, charSpacing: 0.8,
    });
    const hasImage = fs.existsSync(p.image);
    if (hasImage && p.aspect) {
      const h = colW / p.aspect;
      const y = imgTop + (imgH - h);
      s.addImage({
        path: p.image,
        x, y, w: colW, h,
      });
    } else if (hasImage) {
      s.addImage({
        path: p.image,
        x, y: imgTop, w: colW, h: imgH,
        sizing: { type: "contain", w: colW, h: imgH },
      });
    } else {
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: imgTop, w: colW, h: imgH,
        fill: { color: COLOR.cardCream },
        line: { color: COLOR.hairline, width: 0.75 },
      });
      s.addText("Lopata and Govind, JCTC 9, 4939 (2013). The pentanone tune is not the production value. μ and ε₀ are placeholders, retuned for the molecule that is chosen.", {
        x: x + 0.18, y: imgTop + 0.35, w: colW - 0.36, h: imgH - 0.7,
        fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0,
      });
    }
    s.addText(p.cap, {
      x, y: 4.88, w: colW, h: 0.72,
      fontFace: FONT.body, fontSize: 13, color: COLOR.body, margin: 0,
    });
  });

  s.addText("None of these is the molecule in Steps 2–8. MOs 21, 23, and 24 are the pentanone frontier, not the production one. This campaign does not reuse the sodium line or that gold sphere.", {
    x: MARGIN_X, y: 5.75, w: W - 2 * MARGIN_X, h: 0.95,
    fontFace: FONT.body, fontSize: 15, italic: true, color: COLOR.forest, margin: 0,
  });

  s.addNotes("Do not rerun these checks instead of Steps 2–8, and do not overlay their traces on the new jobs. The DCH panel is the Cartesian PySCF sweep against the digitized Nascimento Fig. 8a. Inputs that say driver dch are retired.");
}

// ----------------------------------------------------------------------
// 4  Run list
// ----------------------------------------------------------------------
function slideList() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "THE RUN LIST", "04  /  " + N);
  addTitle(s, "Eight steps. Only the screen has been run.", 0.64, 0.46, 28);

  const rows = [
    ["1", "Literature and screen", "Bright valence lines near a metallic plasmon.", "Done", COLOR.forest],
    ["2", "Core survey", "Which MO is the core on the resonant molecule.", "Placeholder", COLOR.terracotta],
    ["3", "Field-free DCH", "The sudden hole, one factor at a time. D1–D6.", "Placeholder", COLOR.terracotta],
    ["4", "δ-kick spectra", "Isolated absorption, ± DCH, ± CAP. K1–K4.", "Placeholder", COLOR.terracotta],
    ["5", "Gaussian hybrid", "Cross section, ± DCH, ± NP, ± CAP. G1–G8.", "Placeholder", COLOR.terracotta],
    ["6", "Two controls on G4", "Perpendicular, and back-propagation off.", "Placeholder", COLOR.terracotta],
    ["7", "Distance", "Resonant pair at three other surface gaps.", "Placeholder", COLOR.terracotta],
    ["8", "Detuned pair", "Key runs where the resonances do not overlap.", "Placeholder", COLOR.terracotta],
  ];
  rows.forEach((r, i) => {
    const y = 1.2 + i * 0.72;
    s.addText(r[0], {
      x: MARGIN_X, y, w: 0.45, h: 0.55,
      fontFace: FONT.head, fontSize: 20, color: COLOR.forest, margin: 0, valign: "middle",
    });
    s.addText(r[1], {
      x: 1.3, y, w: 3.15, h: 0.55,
      fontFace: FONT.head, fontSize: 16, color: COLOR.ink, margin: 0, valign: "middle",
    });
    s.addText(r[2], {
      x: 4.55, y, w: 6.15, h: 0.55,
      fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0, valign: "middle",
    });
    s.addText(r[3], {
      x: 10.85, y: y + 0.1, w: 1.75, h: 0.36,
      fontFace: FONT.body, fontSize: 12, color: r[4], align: "right", margin: 0, valign: "middle",
    });
    if (i < rows.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: MARGIN_X, y: y + 0.64, w: W - 2 * MARGIN_X, h: 0,
        line: { color: COLOR.hairline, width: 0.75 },
      });
    }
  });

  s.addNotes("There is no separate step for choosing the pair. Choose it, then fill the placeholders, then launch. Steps 2–7 are the resonant pair. Step 7 changes only the gap. Step 8 is the pair whose resonances do not overlap, still at 0.015 μm.");
}

// ----------------------------------------------------------------------
// 5  Step 1
// ----------------------------------------------------------------------
function slideStep1() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "STEP 1   ·   DONE", "05  /  " + N);
  addTitle(s, "The literature molecules miss both plasmons. One candidate does not.", 0.7, 0.95, 28);

  const stats = [
    ["417 nm", "trans-thioindigo, screen μ", "LC-ωPBE at μ = 0.34272, f = 0.29. That μ is not the production value. The root moves if μ changes."],
    ["418.5 nm", "35 nm silver, not selected", "Mie peak of Meep Ag in water. This pair is resonant with itself. It is not written into the JSON."],
    ["None", "of the DCH-literature set", "No bright line in that screen sits on the silver plasmon near 418 nm or the gold plasmon near 540 nm."],
  ];
  stats.forEach((st, i) => {
    const x = MARGIN_X + i * 4.05;
    s.addText(st[0], {
      x, y: 1.9, w: 3.85, h: 0.6,
      fontFace: FONT.head, fontSize: 32, color: COLOR.forest, margin: 0,
    });
    s.addText(st[1], {
      x, y: 2.55, w: 3.85, h: 0.4,
      fontFace: FONT.head, fontSize: 16, color: COLOR.ink, margin: 0,
    });
    s.addText(st[2], {
      x, y: 3.05, w: 3.85, h: 1.35,
      fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X, y: 4.7, w: W - 2 * MARGIN_X, h: 2.15,
    fill: { color: COLOR.panel },
  });
  s.addText("What Step 1 actually is", {
    x: MARGIN_X + 0.3, y: 4.88, w: 11.5, h: 0.35,
    fontFace: FONT.head, fontSize: 16, color: COLOR.ink, margin: 0,
  });
  s.addText("A reading list of double-core-hole papers, plus a valence-absorption screen of those molecules and of trans-thioindigo. Thioindigo is not in that literature. The production initial condition, from Step 3 on, is the pentanone one: a sudden double vacancy on one core MO, neutral orbitals kept. It is not a two-site free-electron-laser spectrum. The thioindigo μ tune has not been run.", {
    x: MARGIN_X + 0.3, y: 5.3, w: 11.3, h: 1.3,
    fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
  });

  s.addNotes("If thioindigo is later selected, match the calculated root at 417 nm, not the benzene maximum. The 418.5 nm number is the Rakic Ag dielectric, not Ag_visible. Meep marks that Palik fit unstable.");
}

// ----------------------------------------------------------------------
// 6  Locked vs open
// ----------------------------------------------------------------------
function slideLocked() {
  const s = pres.addSlide();
  s.background = { color: COLOR.beige };
  addEyebrow(s, "FROM STEP 2 ON", "06  /  " + N);
  addTitle(s, "The form is fixed. μ and ε₀ are not.", 0.7, 0.55, 30);

  s.addText("HELD FIXED", {
    x: MARGIN_X, y: 1.45, w: 5.6, h: 0.28,
    fontFace: FONT.body, fontSize: 12, color: COLOR.forest, margin: 0, charSpacing: 1.4,
  });
  s.addText("STILL A PLACEHOLDER", {
    x: 7.15, y: 1.45, w: 5.4, h: 0.28,
    fontFace: FONT.body, fontSize: 12, color: COLOR.terracotta, margin: 0, charSpacing: 1.4,
  });

  const left = [
    ["Functional", "LC-ωPBE on every run. Not PBE0. The value of μ is open"],
    ["Basis", "6-311G*, Cartesian. Charge 0, singlet. Nuclei stay at PBE0"],
    ["CAP shape", "Static. γ₀ = 1, ξ = 0.5, clamp 100. ε₀ is open"],
    ["Time", "Molecule-only: dt 0.05 au. Hybrid: dt 0.1 au to 10,000 au"],
    ["Cell", "Water, n = 1.33. Baseline gap 0.015 μm. Step 7 is the scan"],
    ["Gaussian width", "fwidth 2.0. The wavelength itself is still open"],
  ];
  const right = [
    ["Geometry", "Resonant file in Steps 2–7. A separate file in Step 8"],
    ["Core and frontier", "One core index, and three watch indices"],
    ["Nanoparticle", "Meep material and radius. Baseline x is radius + 0.015 μm"],
    ["Gaussian wavelength", "The molecule’s bright root, in μm"],
    ["μ", "One number on every file for that molecule, CAP on or off"],
    ["ε₀", "CAP threshold in Ha, tuned at that μ. CAP files only"],
  ];
  left.forEach((row, i) => {
    const y = 1.78 + i * 0.82;
    s.addText(row[0], {
      x: MARGIN_X, y, w: 5.6, h: 0.3,
      fontFace: FONT.head, fontSize: 16, color: COLOR.ink, margin: 0,
    });
    s.addText(row[1], {
      x: MARGIN_X, y: y + 0.28, w: 5.7, h: 0.46,
      fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0,
    });
  });
  right.forEach((row, i) => {
    const y = 1.78 + i * 0.82;
    s.addText(row[0], {
      x: 7.15, y, w: 5.4, h: 0.3,
      fontFace: FONT.head, fontSize: 16, color: COLOR.ink, margin: 0,
    });
    s.addText(row[1], {
      x: 7.15, y: y + 0.28, w: 5.4, h: 0.46,
      fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0,
    });
  });

  s.addNotes("μ and ε₀ are tuned for the chosen molecule. The pentanone pair, 0.34272 and 0.003028 Ha, is not pasted in. A CAP-off job must use the same μ as its CAP-on partner. ε₀ appears only in the CAP block. Do not launch until every PLACEHOLDER_ token in that file is a real value.");
}

// ----------------------------------------------------------------------
// 7  Step 2
// ----------------------------------------------------------------------
function slideStep2() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "STEP 2   ·   PLACEHOLDER", "07  /  " + N);
  addTitle(s, "Name the core before any hole is made.", 0.7, 0.55, 30);

  s.addText("survey.json builds the neutral molecule, writes which atoms contribute to MOs 0 through 5, and exits. It does not remove electrons and it does not propagate. The production hole is two electrons from one of those MOs, so the propagation stays closed-shell.", {
    x: MARGIN_X, y: 1.5, w: 7.3, h: 1.45,
    fontFace: FONT.body, fontSize: 16, color: COLOR.body, margin: 0,
  });

  const facts = [
    ["Print, don’t ionize", "The 0–5 list is what the survey reports. It is not the production hole."],
    ["One MO", "A second index would open the shell. Single core holes are out of this plan."],
    ["Equivalent atoms", "A canonical MO can be a combination of two sites, not a hole on one of them."],
  ];
  facts.forEach((f, i) => {
    const y = 3.15 + i * 1.2;
    s.addText(f[0], {
      x: MARGIN_X, y, w: 7.3, h: 0.32,
      fontFace: FONT.head, fontSize: 18, color: COLOR.ink, margin: 0,
    });
    s.addText(f[1], {
      x: MARGIN_X, y: y + 0.34, w: 7.3, h: 0.6,
      fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.4, y: 1.5, w: 4.2, h: 5.15,
    fill: { color: COLOR.darkBg },
  });
  s.addText("THIOINDIGO TRIAL", {
    x: 8.68, y: 1.75, w: 3.7, h: 0.3,
    fontFace: FONT.body, fontSize: 12, color: COLOR.moss, margin: 0, charSpacing: 1.2,
  });
  s.addText("Not the production assignment.", {
    x: 8.68, y: 2.15, w: 3.7, h: 0.7,
    fontFace: FONT.head, fontSize: 20, color: COLOR.titleOnDark, margin: 0,
  });
  const trial = [
    ["MO 0, 1", "Sulfur 1s"],
    ["MO 2, 3", "Oxygen 1s"],
    ["MO 4, 5", "Carbon 1s"],
  ];
  trial.forEach((t, i) => {
    const y = 3.1 + i * 0.75;
    s.addText(t[0], {
      x: 8.68, y, w: 3.7, h: 0.26,
      fontFace: FONT.body, fontSize: 13, color: COLOR.mutedLight, margin: 0,
    });
    s.addText(t[1], {
      x: 8.68, y: y + 0.26, w: 3.7, h: 0.32,
      fontFace: FONT.head, fontSize: 18, color: COLOR.titleOnDark, margin: 0,
    });
  });
  s.addText("The two oxygen 1s orbitals are a mix of both carbonyls. Do not copy these indices unless thioindigo is the molecule you select.", {
    x: 8.68, y: 5.45, w: 3.65, h: 0.95,
    fontFace: FONT.body, fontSize: 13, color: COLOR.subtitleOnDark, margin: 0,
  });

  s.addNotes("Do not launch Step 3 until the resonant survey has been run and the core index is written into the DCH files. The trial lives in Step_2/thioindigo_trial.");
}

// ----------------------------------------------------------------------
// 8  Step 3
// ----------------------------------------------------------------------
function slideStep3() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "STEP 3   ·   FIELD-FREE", "08  /  " + N);
  addTitle(s, "The drive is the sudden hole, not a laser.", 0.68, 0.5, 28);

  const header = [
    { t: "Run", x: 0.7, w: 0.8 },
    { t: "CAP", x: 1.6, w: 1.1 },
    { t: "Sphere", x: 2.8, w: 1.3 },
    { t: "Radiates", x: 4.2, w: 1.5 },
    { t: "What the difference is", x: 5.9, w: 6.6 },
  ];
  header.forEach((h) => {
    s.addText(h.t, {
      x: h.x, y: 1.35, w: h.w, h: 0.3,
      fontFace: FONT.body, fontSize: 12, color: COLOR.muted, margin: 0, charSpacing: 0.8,
    });
  });
  const rows = [
    ["D1", "off", "no", "—", "The free hole. Stock geometry. Can checkpoint."],
    ["D2", "on", "no", "—", "D2 − D1 is the CAP on that free hole."],
    ["D3", "off", "yes", "on", "D3 − D1 is the sphere when the molecule radiates."],
    ["D4", "off", "yes", "off", "If D4 matches D1, the effect is the radiated field coming back."],
    ["D5", "on", "yes", "on", "D5 − D3 is the CAP on the radiating case."],
    ["D6", "on", "yes", "off", "D6 − D4 is the CAP when the molecule does not source the cell."],
  ];
  rows.forEach((r, i) => {
    const y = 1.75 + i * 0.72;
    const bg = i % 2 === 0 ? COLOR.panel : COLOR.cream;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.55, y, w: 12.25, h: 0.68,
      fill: { color: bg },
    });
    const vals = [r[0], r[1], r[2], r[3]];
    const xs = [0.7, 1.6, 2.8, 4.2];
    const ws = [0.8, 1.1, 1.3, 1.5];
    vals.forEach((v, j) => {
      s.addText(v, {
        x: xs[j], y, w: ws[j], h: 0.68,
        fontFace: j === 0 ? FONT.head : FONT.body,
        fontSize: j === 0 ? 16 : 14,
        color: COLOR.ink, margin: 0, valign: "middle",
      });
    });
    s.addText(r[4], {
      x: 5.9, y, w: 6.7, h: 0.68,
      fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0, valign: "middle",
    });
  });

  s.addText("After D1, paste the rotation that puts the core-hole dipole on +x into D3–D6 only. Compare occupations on the first 400 au. The hybrid step is 0.1 au, not 0.05.", {
    x: MARGIN_X, y: 6.15, w: 11.9, h: 0.85,
    fontFace: FONT.body, fontSize: 14, italic: true, color: COLOR.forest, margin: 0,
  });

  s.addNotes("D3–D6 have no Meep source and cannot checkpoint. Do not treat them as cross sections. Neutral field-free dynamics are not in the matrix: with no hole and no field the density does not move.");
}

// ----------------------------------------------------------------------
// 9  Step 4
// ----------------------------------------------------------------------
function slideStep4() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "STEP 4   ·   δ-KICK", "09  /  " + N);
  addTitle(s, "The real-time spectrum of the molecule alone.", 0.7, 0.55, 30);

  s.addText("Absorption driver, polarization full, so x, y, and z are averaged. Kick 0.001 au. Propagate 4000 au. The Fourier window γ = 0.01 is about 0.27 eV half-width. It is not a substitute for the CAP. The plotted window is 1.5–15 eV unless the bright root sits above that.", {
    x: MARGIN_X, y: 1.45, w: 12.0, h: 1.05,
    fontFace: FONT.body, fontSize: 16, color: COLOR.body, margin: 0,
  });

  const jobs = [
    ["K1", "Neutral, no CAP", "The ordinary spectrum. A rotation would not change the averaged curve."],
    ["K2", "Neutral, CAP on", "K2 − K1 is the CAP on the closed shell."],
    ["K3", "DCH, no CAP", "K3 − K1 is the sudden hole."],
    ["K4", "DCH, CAP on", "K4 − K3 is the CAP on the hole."],
  ];
  jobs.forEach((j, i) => {
    const x = MARGIN_X + (i % 2) * 6.15;
    const y = 2.75 + Math.floor(i / 2) * 1.7;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 5.95, h: 1.52,
      fill: { color: COLOR.cardCream },
      line: { color: COLOR.hairline, width: 0.75 },
    });
    s.addText(j[0], {
      x: x + 0.22, y: y + 0.16, w: 1.1, h: 0.4,
      fontFace: FONT.head, fontSize: 20, color: COLOR.forest, margin: 0,
    });
    s.addText(j[1], {
      x: x + 1.35, y: y + 0.2, w: 4.3, h: 0.36,
      fontFace: FONT.head, fontSize: 16, color: COLOR.ink, margin: 0,
    });
    s.addText(j[2], {
      x: x + 0.22, y: y + 0.7, w: 5.5, h: 0.62,
      fontFace: FONT.body, fontSize: 14, color: COLOR.body, margin: 0,
    });
  });

  s.addText("Step 1’s linear response is the check on the same neutral molecule. It is not a substitute for K1. K1’s three dipole files are a valence axis, not the rotation used for D3–D6.", {
    x: MARGIN_X, y: 6.3, w: 12.0, h: 0.7,
    fontFace: FONT.body, fontSize: 14, italic: true, color: COLOR.forest, margin: 0,
  });

  s.addNotes("These jobs can checkpoint. A core hole writes mo_occ.csv under x_dir, y_dir, and z_dir. Lower gamma before treating the 4000 au trace as higher resolution. The dipole is already small by a few hundred au.");
}

// ----------------------------------------------------------------------
// 10  Step 5
// ----------------------------------------------------------------------
function slideStep5() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "STEP 5   ·   GAUSSIAN HYBRID", "10  /  " + N);
  addTitle(s, "The same pulse, with and without the sphere.", 0.68, 0.5, 28);

  s.addText("The Gaussian is centered on the molecule’s bright root, fwidth 2.0. Parallel puts the field along the nanoparticle–molecule axis. Single is the empty cell. Back-propagation is on. The published curve is cross_section. No checkpoints.", {
    x: MARGIN_X, y: 1.32, w: 12.0, h: 0.75,
    fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
  });

  // 2 x 4 grid of G jobs: rows = CAP off/on, conceptually two groups
  const jobs = [
    ["G1", "neutral", "no sphere", "no CAP"],
    ["G2", "DCH", "no sphere", "no CAP"],
    ["G3", "neutral", "sphere", "no CAP"],
    ["G4", "DCH", "sphere", "no CAP"],
    ["G5", "neutral", "no sphere", "CAP"],
    ["G6", "DCH", "no sphere", "CAP"],
    ["G7", "neutral", "sphere", "CAP"],
    ["G8", "DCH", "sphere", "CAP"],
  ];
  jobs.forEach((j, i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x = MARGIN_X + col * 3.05;
    const y = 2.2 + row * 1.55;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.92, h: 1.4,
      fill: { color: row === 0 ? COLOR.cardCream : COLOR.panel },
      line: { color: COLOR.hairline, width: 0.75 },
    });
    s.addText(j[0], {
      x: x + 0.16, y: y + 0.12, w: 2.6, h: 0.34,
      fontFace: FONT.head, fontSize: 18, color: COLOR.forest, margin: 0,
    });
    s.addText(j[1] + "   ·   " + j[2], {
      x: x + 0.16, y: y + 0.5, w: 2.6, h: 0.3,
      fontFace: FONT.body, fontSize: 13, color: COLOR.ink, margin: 0,
    });
    s.addText(j[3], {
      x: x + 0.16, y: y + 0.84, w: 2.6, h: 0.32,
      fontFace: FONT.body, fontSize: 13, color: COLOR.body, margin: 0,
    });
  });

  s.addText("Run G1–G4 first. G4 − G2 is the sphere at fixed DCH. G2 − G1 is the hole with no sphere. G8 − G4 is the CAP on the production hybrid. The empty cell is not a substitute for Step 4.", {
    x: MARGIN_X, y: 5.5, w: 12.0, h: 1.4,
    fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
  });

  s.addNotes("Do not reuse a vacuum reference computed with a different source or a different dt. The plot window is a placeholder until the bright root is known. gamma is 0 on these jobs.");
}

// ----------------------------------------------------------------------
// 11  Steps 6 and 7
// ----------------------------------------------------------------------
function slideLate() {
  const s = pres.addSlide();
  s.background = { color: COLOR.beige };
  addEyebrow(s, "AFTER G4 MOVES", "11  /  " + N);
  addTitle(s, "Two controls at 0.015 μm, then the gap.", 0.68, 0.55, 28);

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X, y: 1.5, w: 5.85, h: 5.3,
    fill: { color: COLOR.cardCream },
    line: { color: COLOR.hairline, width: 0.75 },
  });
  s.addText("STEP 6", {
    x: 0.95, y: 1.7, w: 5.3, h: 0.26,
    fontFace: FONT.body, fontSize: 12, color: COLOR.terracotta, margin: 0, charSpacing: 1.3,
  });
  s.addText("Only after resonant G4 moves.", {
    x: 0.95, y: 2.05, w: 5.3, h: 0.7,
    fontFace: FONT.head, fontSize: 22, color: COLOR.ink, margin: 0,
  });
  s.addText("G4 perpendicular\nPolarization perpendicular instead of parallel.", {
    x: 0.95, y: 2.9, w: 5.3, h: 0.75,
    fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
  });
  s.addText("G4, no back-propagation\nThe molecule feels the cell and does not source it.", {
    x: 0.95, y: 3.75, w: 5.3, h: 0.8,
    fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
  });
  s.addText("One of each is enough. Do not repeat them across CAP, DCH, or the Step 7 gaps until this G4 has moved.", {
    x: 0.95, y: 4.85, w: 5.3, h: 1.5,
    fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.8, y: 1.5, w: 5.85, h: 5.3,
    fill: { color: COLOR.darkBg },
  });
  s.addText("STEP 7", {
    x: 7.05, y: 1.7, w: 5.35, h: 0.26,
    fontFace: FONT.body, fontSize: 12, color: COLOR.moss, margin: 0, charSpacing: 1.3,
  });
  s.addText("Same pair. Three other gaps.", {
    x: 7.05, y: 2.05, w: 5.35, h: 0.7,
    fontFace: FONT.head, fontSize: 22, color: COLOR.titleOnDark, margin: 0,
  });
  s.addText("0.005, 0.030, and 0.060 μm. The 0.015 μm baseline is not repeated. Each gap is D3, D4, and G4 only.", {
    x: 7.05, y: 2.9, w: 5.35, h: 1.05,
    fontFace: FONT.body, fontSize: 15, color: COLOR.titleOnDark, margin: 0,
  });
  s.addText("Compare D3 − D1 and G4 − G2 with the baseline. If the gap changes D3 and G4 but not D4, the distance dependence is the field coming back.", {
    x: 7.05, y: 4.05, w: 5.35, h: 1.15,
    fontFace: FONT.body, fontSize: 15, color: COLOR.subtitleOnDark, margin: 0,
  });
  s.addText("Molecule x is the radius plus that gap, not the baseline x. Widen the cell if the farther gap no longer fits.", {
    x: 7.05, y: 5.35, w: 5.35, h: 1.05,
    fontFace: FONT.body, fontSize: 15, color: COLOR.subtitleOnDark, margin: 0,
  });

  s.addNotes("Step 7 does not repeat molecule-only jobs. D1 and G2 have no nanoparticle. No CAP and no perpendicular run. The detuned pair stays at 0.015 μm so distance and resonance are not changed together.");
}

// ----------------------------------------------------------------------
// 12  Step 8
// ----------------------------------------------------------------------
function slideDetuned() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "STEP 8   ·   OFF RESONANCE", "12  /  " + N);
  addTitle(s, "The same question, where the plasmon misses the root.", 0.7, 0.6, 28);

  s.addText("The Gaussian stays on the molecule’s bright root. The gap stays 0.015 μm. This comparison is resonance, not distance.", {
    x: MARGIN_X, y: 1.5, w: 12.0, h: 0.55,
    fontFace: FONT.body, fontSize: 16, color: COLOR.body, margin: 0,
  });

  const jobs = [
    ["Survey", "The core index belongs to this molecule."],
    ["D1, D3, D4", "Free hole, then the sphere with and without radiation."],
    ["K1, K3", "Where the bright root is, before and after the sudden hole."],
    ["G1–G4", "Cross section, ± DCH, ± nanoparticle, no CAP."],
  ];
  jobs.forEach((j, i) => {
    const x = MARGIN_X + (i % 2) * 6.15;
    const y = 2.25 + Math.floor(i / 2) * 1.55;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 5.95, h: 1.38,
      fill: { color: COLOR.cardCream },
      line: { color: COLOR.hairline, width: 0.75 },
    });
    s.addText(j[0], {
      x: x + 0.24, y: y + 0.16, w: 5.45, h: 0.38,
      fontFace: FONT.head, fontSize: 18, color: COLOR.ink, margin: 0,
    });
    s.addText(j[1], {
      x: x + 0.24, y: y + 0.62, w: 5.45, h: 0.55,
      fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X, y: 5.55, w: W - 2 * MARGIN_X, h: 1.35,
    fill: { color: COLOR.darkBg },
  });
  s.addText("Read D3 − D1 and G4 − G2 against the resonant pair. If the nanoparticle effect shrinks here, it was the shared resonance. The detuned tokens are a separate set.", {
    x: MARGIN_X + 0.3, y: 5.72, w: W - 2 * MARGIN_X - 0.6, h: 1.0,
    fontFace: FONT.head, fontSize: 16, italic: true, color: COLOR.titleOnDark, margin: 0,
  });

  s.addNotes("K1 is what makes detuned a measurement. Do not repeat the CAP rows, the Step 6 controls, or the Step 7 gaps. If only the nanoparticle changes, reuse D1, K1, K3, G1, and G2.");
}

// ----------------------------------------------------------------------
// 13  Still open
// ----------------------------------------------------------------------
function slideOpen() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  addEyebrow(s, "STILL OPEN", "13  /  " + N);
  addTitle(s, "Nothing from Step 2 on should be launched yet.", 0.7, 0.55, 28);

  const items = [
    ["01", "Choose both pairs", "A resonant pair, plasmon on the bright root. A detuned pair, plasmon off that root. Then replace every placeholder."],
    ["02", "Survey, then the index", "Write the core MO and the three frontier indices into the resonant DCH files. Do not reuse the thioindigo trial unless that molecule is selected."],
    ["03", "D1, then the rotation", "Paste the orientation that puts the moving core-hole dipole on +x into D3–D6 only. Leave D1 and D2 on the stock frame."],
    ["04", "G1–G4 before the rest", "The CAP rows, the G4 controls, and the gap scan wait on that 2×2. Step 8 waits until the resonant comparison is in hand."],
  ];
  items.forEach((it, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = MARGIN_X + col * 6.15;
    const y = 1.55 + row * 2.15;
    s.addText(it[0], {
      x, y, w: 0.7, h: 0.4,
      fontFace: FONT.head, fontSize: 18, color: COLOR.terracotta, margin: 0,
    });
    s.addText(it[1], {
      x: x + 0.8, y, w: 5.0, h: 0.4,
      fontFace: FONT.head, fontSize: 18, color: COLOR.ink, margin: 0,
    });
    s.addText(it[2], {
      x, y: y + 0.5, w: 5.85, h: 1.15,
      fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: MARGIN_X, y: 6.0, w: W - 2 * MARGIN_X, h: 0.95,
    fill: { color: COLOR.darkBg },
  });
  s.addText("A file that still contains PLACEHOLDER_ is not an input. The Step 7 x tokens are radius plus that gap. Detuned tokens are only in Step 8.", {
    x: MARGIN_X + 0.3, y: 6.12, w: W - 2 * MARGIN_X - 0.6, h: 0.72,
    fontFace: FONT.head, fontSize: 16, italic: true, color: COLOR.titleOnDark, margin: 0, valign: "middle",
  });

  s.addNotes("Step 7 runs only if resonant G4 − G2 is nonzero. Step 8 stays at 0.015 μm. Hybrid jobs cannot checkpoint. Publish peak-normalized curves when the claim is that a peak moved or broadened.");
}

slideCover();
slideQuestion();
slideChecked();
slideList();
slideStep1();
slideLocked();
slideStep2();
slideStep3();
slideStep4();
slideStep5();
slideLate();
slideDetuned();
slideOpen();

pres.writeFile({ fileName: "/Users/bldrdge1/Downloads/repos/PlasMol/jobs/final/research_plan.pptx" })
  .then(() => console.log("wrote research_plan.pptx"))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
