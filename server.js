/**
 * ClasSync - Groq Vision Server (ENHANCED with Quality Control)
 *
 * Features:
 * - Image preprocessing (contrast, sharpness, upscaling) for curriculum scans
 * - Quality validation before showing results
 * - Curriculum reference hydration for 100% accurate data
 * - Summer semester awareness (between 3rd and 4th year)
 *
 * Setup:
 *   1. Go to https://console.groq.com and sign up (Google login works)
 *   2. Go to https://console.groq.com/keys and create an API key
 *   3. Copy .env.example to .env and paste your key
 *   4. npm install
 *   5. node server.js
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const OpenAI = require("openai");
const fs = require("fs");
const sharp = require("sharp");
const { CURRICULUM_REFERENCE } = require("./curriculum-reference");

const app = express();
const PORT = process.env.PORT || 3001;

// Configure multer for image uploads
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Only JPG, PNG, and WEBP images are allowed"));
    }
  },
});

app.use(cors());
app.use(express.json());

// Initialize Groq client (OpenAI-compatible)
const groq = new OpenAI({
  baseURL: "https://api.groq.com/openai/v1",
  apiKey: process.env.GROQ_API_KEY,
});

// ==================== IMAGE PREPROCESSING ====================
async function preprocessImage(filePath) {
  console.log("   🔧 Preprocessing image...");

  try {
    const image = sharp(filePath);
    const metadata = await image.metadata();

    console.log(`   📐 Original: ${metadata.width}x${metadata.height}`);

    // Determine if upscaling is needed
    const minWidth = 1200;
    const needsUpscale = metadata.width < minWidth;

    let processedImage = image;

    // 1. Convert to grayscale for better text recognition
    processedImage = processedImage.grayscale();

    // 2. Upscale if resolution is too low
    if (needsUpscale) {
      const scale = Math.ceil(minWidth / metadata.width);
      processedImage = processedImage.resize(
        metadata.width * scale,
        metadata.height * scale,
        { kernel: "lanczos3" }
      );
      console.log(`   ⬆️  Upscaled ${scale}x`);
    }

    // 3. Enhance contrast and sharpness
    processedImage = processedImage
      .normalize() // Auto-adjust contrast
      .sharpen({ sigma: 1.5 }) // Sharpen text
      .linear(1.2, -(128 * 0.2)); // Increase contrast slightly

    // 4. Save processed image
    const processedPath = filePath + "_processed.png";
    await processedImage.toFile(processedPath);

    console.log("   ✅ Preprocessing complete");
    return processedPath;
  } catch (error) {
    console.log(`   ⚠️  Preprocessing failed: ${error.message}`);
    return filePath; // Return original if preprocessing fails
  }
}

// ==================== QUALITY ANALYSIS ====================
async function analyzeImageQuality(filePath) {
  try {
    const image = sharp(filePath);
    const metadata = await image.metadata();
    const stats = await image.stats();

    const quality = {
      width: metadata.width,
      height: metadata.height,
      resolution: metadata.width * metadata.height,
      isLowRes: metadata.width < 800 || metadata.height < 600,
      isBlurry: false,
      brightness: stats.channels[0].mean,
      isDark: stats.channels[0].mean < 60,
      isBright: stats.channels[0].mean > 200,
    };

    return quality;
  } catch (error) {
    return { error: error.message, isLowRes: true };
  }
}

// ==================== EXTRACTION QUALITY VALIDATION ====================
function validateExtractionQuality(subjects, imageNumber) {
  const validation = {
    isGood: false,
    issues: [],
    warnings: [],
    subjectCount: subjects.length,
    missingCourses: [],
  };

  // Expected counts per image
  const expectedCounts = {
    1: { min: 27, max: 31, description: "1st & 2nd Year" },
    2: { min: 23, max: 27, description: "3rd Year + Summer + 4th Year" },
  };

  const expected = expectedCounts[imageNumber];

  if (expected) {
    if (subjects.length < expected.min) {
      validation.issues.push(
        `Only ${subjects.length} subjects found (expected ${expected.min}-${expected.max} for ${expected.description})`
      );
    } else if (subjects.length > expected.max) {
      validation.warnings.push(
        `Found ${subjects.length} subjects (expected ${expected.min}-${expected.max}). May have duplicates.`
      );
    } else {
      validation.isGood = true;
    }
  }

  // Check year distribution
  const byYear = {};
  subjects.forEach((s) => {
    byYear[s.yearLevel] = (byYear[s.yearLevel] || 0) + 1;
  });

  // Image-specific checks
  if (imageNumber === 1) {
    if (!byYear["1st Year"] || byYear["1st Year"] < 12) {
      validation.issues.push("1st Year incomplete or missing");
    }
    if (!byYear["2nd Year"] || byYear["2nd Year"] < 13) {
      validation.issues.push("2nd Year incomplete or missing");
    }
  }

  if (imageNumber === 2) {
    if (!byYear["3rd Year"] || byYear["3rd Year"] < 11) {
      validation.issues.push("3rd Year incomplete or missing");
    }
    if (!byYear["4th Year"] || byYear["4th Year"] < 9) {
      validation.issues.push("4th Year incomplete or missing");
    }

    // Check for Summer (CS 122)
    const hasSummer = subjects.some((s) => s.subjectCode === "CS 122");
    if (!hasSummer) {
      validation.issues.push("Summer semester missing (CS 122 - Practicum)");
      validation.missingCourses.push("CS 122");
    } else {
      const cs122 = subjects.find((s) => s.subjectCode === "CS 122");
      if (cs122.semester !== "Summer") {
        validation.warnings.push(
          `CS 122 in wrong semester (${cs122.semester}), should be Summer`
        );
      }
    }
  }

  validation.isGood = validation.issues.length === 0;

  return validation;
}

// ==================== OVERALL VALIDATION ====================
function validateOverallExtraction(allSubjects, imageResults) {
  const validation = {
    success: false,
    quality: "poor",
    totalCount: allSubjects.length,
    issues: [],
    warnings: [],
    missingCourses: [],
    byYear: {},
  };

  // Count by year
  allSubjects.forEach((s) => {
    validation.byYear[s.yearLevel] =
      (validation.byYear[s.yearLevel] || 0) + 1;
  });

  // Check if individual images are good
  const image1Good = imageResults[0]?.validation?.isGood || false;
  const image2Good = imageResults[1]?.validation?.isGood || false;
  const bothImagesGood = image1Good && image2Good;

  // Total count check
  if (allSubjects.length >= 54) {
    validation.quality = "excellent";
    validation.success = true;
  } else if (allSubjects.length >= 50) {
    validation.quality = "good";
    validation.success = true;
    validation.warnings.push(`${54 - allSubjects.length} subjects missing`);
  } else if (allSubjects.length >= 45) {
    validation.quality = "moderate";
    if (
      bothImagesGood ||
      (imageResults.length === 1 && image1Good)
    ) {
      validation.success = true;
      validation.warnings.push(
        `Only ${allSubjects.length}/54 subjects total, but image quality is good`
      );
    } else {
      validation.issues.push(
        `Only ${allSubjects.length}/54 subjects extracted`
      );
    }
  } else {
    validation.quality = "poor";
    if (
      imageResults.length === 1 &&
      (image1Good || image2Good)
    ) {
      validation.quality = "moderate";
      validation.success = true;
      validation.warnings.push(
        `Partial scan: ${allSubjects.length} subjects extracted from 1 image`
      );
    } else {
      validation.issues.push(
        `Very poor extraction: only ${allSubjects.length}/54 subjects`
      );
    }
  }

  // Check all years present (only if expecting full scan - 2 images)
  if (imageResults.length === 2) {
    const expectedYears = ["1st Year", "2nd Year", "3rd Year", "4th Year"];
    expectedYears.forEach((year) => {
      if (!validation.byYear[year] || validation.byYear[year] === 0) {
        validation.issues.push(`${year} completely missing`);
      }
    });
  }

  // Check for Summer
  const hasSummer = allSubjects.some((s) => s.subjectCode === "CS 122");
  if (!hasSummer && imageResults.length === 2) {
    validation.issues.push("Summer semester missing (CS 122)");
    validation.missingCourses.push("CS 122");
  }

  // Find specific missing courses
  const extractedCodes = new Set(allSubjects.map((s) => s.subjectCode));
  const allReferenceCodes = Object.keys(CURRICULUM_REFERENCE);
  validation.missingCourses = allReferenceCodes.filter(
    (code) => !extractedCodes.has(code)
  );

  return validation;
}

// ==================== HYDRATE SUBJECTS FROM REFERENCE ====================
function hydrateSubjectsFromReference(subjects) {
  // Build a lookup map: normalized key → { refKey, reference }
  const refLookup = {};
  Object.keys(CURRICULUM_REFERENCE).forEach((key) => {
    // Normalize: uppercase, remove ALL spaces
    const normalized = key.toUpperCase().replace(/\s+/g, "");
    refLookup[normalized] = { refKey: key, reference: CURRICULUM_REFERENCE[key] };
  });

  return subjects.map((subject) => {
    let code = subject.subjectCode.trim();

    // Normalize extracted code the same way: uppercase, remove ALL spaces
    const normalizedCode = code.toUpperCase().replace(/\s+/g, "");

    // Look up in reference
    const match = refLookup[normalizedCode];

    if (match) {
      return {
        subjectCode: match.refKey, // Use the correct formatted key
        subjectName: match.reference.name,
        lecUnits: match.reference.lecUnits,
        labUnits: match.reference.labUnits,
        units: match.reference.units,
        yearLevel: match.reference.yearLevel,
        semester: match.reference.semester,
      };
    }

    // If not in reference, return as-is
    return subject;
  });
}

// ==================== CURRICULUM EXTRACTION PROMPT (SIMPLIFIED) ====================
const CURRICULUM_PROMPT = `You are a document scanner for Bicol University BSCS curriculum.

Extract ALL subject codes visible in this document. Focus on ACCURACY and COMPLETENESS.

INSTRUCTIONS:
1. Extract EVERY subject code across all years (expect 25-30 subjects per image)
2. Scan ALL sections systematically
3. Each year has TWO columns: First Semester (left), Second Semester (right)
4. Third year may have a SUMMER section (separate) with CS 122
5. Skip only the "Total" row at the bottom of each semester

SUBJECT CODE PATTERNS:
- CS courses: CS 101-126
- Math: Math 101, Math 102, Math Elec 101, Math Elec 102
- GEC: GEC 11-20, GEC Elec 1, GEC Elec 2, GEC Elec 21, GEC Elec 22
- CS Electives: CS Elec 1-3
- Other: Phys 1, PATHFIT 1-4, NSTP 11, NSTP 2

IMPORTANT:
- Extract the code EXACTLY as shown (preserve spacing)
- Detect year and semester from document layout
- If you see a separate "SUMMER" section, mark CS 122 as semester: "Summer"

Return ONLY valid JSON:
{
  "subjects": [
    {"subjectCode": "CS 101", "subjectName": "", "lecUnits": 0, "labUnits": 0, "units": 0, "yearLevel": "1st Year", "semester": "1st Semester"}
  ],
  "documentType": "curriculum_checklist",
  "totalSubjectsFound": 29,
  "confidence": "high"
}`;

// ==================== COR EXTRACTION PROMPT ====================
const COR_PROMPT = `You are a document scanner for a Philippine university (Bicol University).
You are looking at a Certificate of Registration (COR) document.

A COR typically contains:
- Student name, student number, program/course
- The current semester and academic year
- A SCHEDULE table with columns: Code, Subject, Units, Class, Days, Time, Room, Faculty

Extract the student's PROGRAM and ALL enrolled subjects/courses from this COR.

For each subject/course, extract these fields:
- subjectCode: The course code (e.g., "CS 125", "GEC 19", "GEC Elect 21.3")
- subjectName: The full course title/description
- units: Total credit units (number)
- schedules: An array of schedule objects, each with:
  - days: Single day code (e.g., "M", "T", "W", "Th", "F")
  - time: The time value (e.g., "01:00 PM - 04:00 PM")
  - room: The room value (e.g., "L1", "CSD 25", "CSD 24")
- section: The class/section name (e.g., "BSCS-P-4A")
- instructor: Faculty/professor name (e.g., "ARISPE, M.", "ALMONTE, R.")

IMPORTANT RULES:
1. Extract EVERY subject listed in the schedule table, do not skip any
2. The table has separate columns for Days, Time, and Room — extract each SEPARATELY
3. Room values are typically short codes like "L1", "CSD 25", "CSD 24", "GYM", etc.
4. The program is near the top (e.g., "Bachelor of Science in Computer Science")
5. If a field is not visible or unclear, set it to ""
6. Units column may show "3.0 3.0 0.0" meaning total=3, lec=3, lab=0 — just use the first number
7. Return ONLY valid JSON — no markdown, no backticks, no explanation

MULTI-DAY SCHEDULES:
Some subjects meet on multiple days (e.g., "MW" = Monday & Wednesday, "TTh" = Tuesday & Thursday).
When a subject has multi-day schedule, return MULTIPLE entries in the "schedules" array — one per day.
- "MW" → two entries: one for "M" and one for "W" (same time, same room)
- "TTh" → two entries: one for "T" and one for "Th"
- "MWF" → three entries: "M", "W", "F"
- Single day like "W", "F", "Th" → one entry

Return this exact format:
{
  "program": "BS Computer Science",
  "courses": [
    {
      "subjectCode": "CS 125",
      "subjectName": "CS Thesis 2",
      "units": 3,
      "schedules": [
        { "days": "W", "time": "01:00 PM - 04:00 PM", "room": "L1" }
      ],
      "section": "BSCS-P-4A",
      "instructor": "ARISPE, M."
    },
    {
      "subjectCode": "MATH 101",
      "subjectName": "Mathematics in the Modern World",
      "units": 3,
      "schedules": [
        { "days": "T", "time": "09:00 AM - 10:30 AM", "room": "CSD 25" },
        { "days": "Th", "time": "09:00 AM - 10:30 AM", "room": "CSD 25" }
      ],
      "section": "BSCS-P-4A",
      "instructor": "SANTOS, J."
    }
  ],
  "totalCoursesFound": 5,
  "confidence": "high"
}`;

// ==================== TIMETABLE EXTRACTION PROMPT ====================
const TIMETABLE_PROMPT = `You are a document scanner for a Philippine university (Bicol University).
You are looking at a student TIMETABLE / CLASS SCHEDULE document from the BU Student Portal.

The timetable is a weekly grid showing:
- Days of the week as columns (Monday through Sunday)
- Time slots as rows (30-minute increments)
- Colored blocks representing classes with subject name, section, room, and instructor

The document header contains:
- Student name
- Academic Year and Semester (e.g., "AY 2021-2022 1st Semester")

Extract the ACADEMIC YEAR, SEMESTER, and ALL subjects/classes from this timetable.

For each subject/class, extract:
- subjectName: The full course title (e.g., "Introduction to Computing", "Computer Programming 1")
- section: The section code (e.g., "PC-BSIT1B", "BSCS-P-4A")
- room: The room/venue (e.g., "Gym 51", "CSD 25", "NSTP Rm. 3")
- instructor: Faculty name (e.g., "Jorge Sulpicio S. Aganan", "Andy Nopre")
- schedules: An array of schedule objects for each day/time the class meets:
  - day: Full day name (e.g., "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
  - startTime: Start time (e.g., "9:30 AM", "1:00 PM")
  - endTime: End time (e.g., "11:00 AM", "2:30 PM")

IMPORTANT RULES:
1. Extract EVERY class block visible in the timetable grid
2. A subject may appear on MULTIPLE days — create a separate schedule entry for each day
3. A subject may appear multiple times on the same day at different times — capture each occurrence
4. Read the time from the row positions (the grid is in 30-minute increments starting from 6:00 AM)
5. The section and room are usually shown below the subject name in each block
6. The instructor name is usually shown below the section/room
7. Academic Year format: "AY XXXX-XXXX" (e.g., "AY 2021-2022")
8. Semester: "1st Semester", "2nd Semester", or "Summer"
9. If a subject code is visible, include it. If only the subject name is shown, set subjectCode to ""
10. Return ONLY valid JSON — no markdown, no backticks, no explanation

Return this exact format:
{
  "academicYear": "AY 2021-2022",
  "semester": "1st Semester",
  "subjects": [
    {
      "subjectName": "Introduction to Computing",
      "subjectCode": "",
      "section": "PC-BSIT1B",
      "room": "Gym 51",
      "instructor": "Jorge Sulpicio S. Aganan",
      "schedules": [
        { "day": "Wednesday", "startTime": "10:30 AM", "endTime": "12:00 PM" },
        { "day": "Tuesday", "startTime": "3:00 PM", "endTime": "4:30 PM" }
      ]
    },
    {
      "subjectName": "Computer Programming 1",
      "subjectCode": "",
      "section": "PC-BSIT1B",
      "room": "Gym 51",
      "instructor": "Andy Nopre",
      "schedules": [
        { "day": "Monday", "startTime": "6:00 PM", "endTime": "7:30 PM" },
        { "day": "Friday", "startTime": "4:30 PM", "endTime": "6:00 PM" }
      ]
    }
  ],
  "totalSubjectsFound": 8,
  "confidence": "high"
}`;

// ==================== GRADE EXTRACTION PROMPT ====================
const GRADES_PROMPT = `You are a document scanner for a Philippine university (Bicol University).
You are looking at a screenshot or photo of a student's grade report/portal.

Extract ALL subject codes and their corresponding grades from this image.

For each subject, extract:
- subjectCode: The course code (e.g., "CS 125", "GEC 19", "GEC Elect 21.3", "MATH 101")
- subjectName: The full course title/description (e.g., "Design and Analysis of Algorithms")
- grade: The numerical grade value (e.g., 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 5.0)

IMPORTANT RULES:
1. Extract EVERY subject and grade pair visible in the image
2. Grades in Philippine universities typically range from 1.0 (highest) to 5.0 (failing)
3. Common grade values: 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0
4. INC = Incomplete, DRP = Dropped — skip these, only extract numeric grades
5. Look for table rows with subject codes paired with grade values
6. Subject codes may appear as: "CS 125", "GEC Elect 21.3", "PATHFIT 1", "NSTP 1", etc.
7. CRITICAL: Do NOT confuse the "Units" columns (Total, Lec, Lab) with the actual grade!
   - Units columns typically show values like 3.0, 5.0, 2.0 and appear BEFORE the grade
   - The GRADE column is usually the LAST numeric column, often near a "PASSED/FAILED" status
   - Example row: "GEC 18 | Ethics | 3.0 | 3.0 | 0.0 | 1.8 | PASSED" → grade is 1.8, NOT 3.0
8. Return ONLY valid JSON — no markdown, no backticks, no explanation

Return this exact format:
{
  "grades": [
    { "subjectCode": "CS 125", "subjectName": "CS Thesis 2", "grade": 1.5 },
    { "subjectCode": "GEC 19", "subjectName": "Ethics", "grade": 1.75 }
  ],
  "totalFound": 5,
  "confidence": "high"
}`;

// ==================== HELPER: Send image to Groq ====================
async function sendToGroq(file, prompt) {
  const imageData = fs.readFileSync(file.path);
  const base64Image = imageData.toString("base64");
  const mimeType = file.mimetype;

  const completion = await groq.chat.completions.create({
    model: "meta-llama/llama-4-scout-17b-16e-instruct",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          {
            type: "image_url",
            image_url: {
              url: `data:${mimeType};base64,${base64Image}`,
            },
          },
        ],
      },
    ],
    temperature: 0,
    max_tokens: 8192,
  });

  const text = completion.choices[0]?.message?.content || "";

  // Parse JSON from response
  try {
    return JSON.parse(text);
  } catch {
    const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[1].trim());
    }
    const objectMatch = text.match(/\{[\s\S]*\}/);
    if (objectMatch) {
      return JSON.parse(objectMatch[0]);
    }
    throw new Error("Could not parse AI response as JSON");
  }
}

// ==================== HELPER: Handle Groq errors ====================
function getErrorMessage(error) {
  if (error.status === 429) {
    return "Rate limited. Wait 1 minute and try again (max 30 req/min).";
  }
  if (error.status === 401) {
    return "Invalid API key. Check your .env file.";
  }
  return error.message;
}

// ==================== SCAN COR ENDPOINT ====================
app.post("/api/scan-cor", upload.single("image"), async (req, res) => {
  console.log("\n📋 Received COR scan request");

  if (!req.file) {
    return res.status(400).json({ success: false, error: "No image uploaded" });
  }

  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({
      success: false,
      error: "GROQ_API_KEY not set. Add it to your .env file.",
    });
  }

  const file = req.file;
  console.log(
    `📎 Image: ${file.originalname} (${(file.size / 1024).toFixed(1)}KB)`,
  );
  console.log("🤖 Sending to Groq (Llama 4 Scout) for COR extraction...");

  try {
    const parsed = await sendToGroq(file, COR_PROMPT);

    const program = parsed.program || "BS Computer Science";
    const courses = (parsed.courses || []).map((c, idx) => ({
      subjectCode: c.subjectCode || `UNKNOWN_${idx + 1}`,
      subjectName: c.subjectName || "Unknown Subject",
      units: parseInt(c.units) || 0,
      schedules: (c.schedules || []).map((s) => ({
        days: s.days || "",
        time: s.time || "",
        room: s.room || "",
      })),
      section: c.section || "",
      instructor: c.instructor || "",
    }));

    console.log(
      `✅ Extracted ${courses.length} courses for program: ${program}`,
    );

    res.json({
      success: true,
      data: {
        program,
        courses,
        totalCoursesFound: courses.length,
        confidence: parsed.confidence || "unknown",
      },
    });
  } catch (error) {
    console.error("❌ COR scan error:", error.message);
    res.status(error.status || 500).json({
      success: false,
      error: getErrorMessage(error),
    });
  } finally {
    try {
      fs.unlinkSync(file.path);
    } catch {}
  }
});

// ==================== SCAN CURRICULUM ENDPOINT (ENHANCED) ====================
app.post("/api/scan-curriculum", upload.array("image", 2), async (req, res) => {
  console.log("\n📸 Received curriculum scan request");

  if (!req.files || req.files.length === 0) {
    return res.status(400).json({
      success: false,
      error: "No image(s) uploaded",
      quality: "error",
    });
  }

  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({
      success: false,
      error: "GROQ_API_KEY not set",
      quality: "error",
    });
  }

  const files = req.files;
  const imageResults = [];
  let allSubjects = [];

  try {
    // Process each image
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const imageNum = i + 1;

      console.log(`\n${"=".repeat(90)}`);
      console.log(
        `📎 Processing Image ${imageNum}/${files.length}: ${file.originalname}`
      );
      console.log(`   Size: ${(file.size / 1024).toFixed(1)}KB`);

      const imageResult = {
        imageNumber: imageNum,
        originalFile: file.originalname,
        subjects: [],
        validation: null,
        quality: null,
      };

      try {
        // Step 1: Analyze original quality
        console.log("   🔍 Analyzing image quality...");
        const quality = await analyzeImageQuality(file.path);
        imageResult.quality = quality;

        if (quality.isLowRes) {
          console.log(
            `   ⚠️  Low resolution detected: ${quality.width}x${quality.height}`
          );
        }
        if (quality.isDark) {
          console.log("   ⚠️  Image appears dark");
        }

        // Step 2: Preprocess image
        const processedPath = await preprocessImage(file.path);

        // Step 3: Send to Groq
        console.log("   🤖 Sending to Groq for extraction...");
        const parsed = await sendToGroq(
          { ...file, path: processedPath },
          CURRICULUM_PROMPT
        );

        if (parsed.subjects && Array.isArray(parsed.subjects)) {
          const cleaned = parsed.subjects.map((s, idx) => ({
            subjectCode: s.subjectCode || `UNKNOWN_${idx + 1}`,
            subjectName: s.subjectName || "",
            lecUnits: parseInt(s.lecUnits) || 0,
            labUnits: parseInt(s.labUnits) || 0,
            units: parseInt(s.units) || 0,
            yearLevel: s.yearLevel || "Unknown",
            semester: s.semester || "Unknown",
          }));

          // Step 4: Hydrate from reference
          const hydrated = hydrateSubjectsFromReference(cleaned);
          imageResult.subjects = hydrated;
          allSubjects.push(...hydrated);

          console.log(`   📊 Extracted: ${hydrated.length} subjects`);
          console.log(
            `   ✨ Hydrated: ${hydrated.filter((s, idx) => s.subjectName !== cleaned[idx].subjectName).length} from reference`
          );

          // Step 5: Validate extraction quality
          const validation = validateExtractionQuality(hydrated, imageNum);
          imageResult.validation = validation;

          if (validation.isGood) {
            console.log(`   ✅ Quality: GOOD`);
          } else {
            console.log(`   ⚠️  Quality: POOR`);
            validation.issues.forEach((issue) =>
              console.log(`      ❌ ${issue}`)
            );
          }
        }

        // Clean up processed image
        if (processedPath !== file.path) {
          try {
            fs.unlinkSync(processedPath);
          } catch {}
        }
      } catch (parseErr) {
        console.log(`   ❌ Extraction failed: ${parseErr.message}`);
        imageResult.validation = {
          isGood: false,
          issues: [parseErr.message],
          warnings: [],
          subjectCount: 0,
        };
      }

      imageResults.push(imageResult);
    }

    // Deduplicate
    const seen = new Set();
    const uniqueSubjects = allSubjects.filter((s) => {
      const key = s.subjectCode.toLowerCase().replace(/\s+/g, "");
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    // Overall validation
    const overallValidation = validateOverallExtraction(
      uniqueSubjects,
      imageResults
    );

    console.log(`\n${"=".repeat(90)}`);
    console.log(`📊 FINAL RESULTS:`);
    console.log(`   Total: ${uniqueSubjects.length}/54 subjects`);
    console.log(`   Quality: ${overallValidation.quality.toUpperCase()}`);

    if (overallValidation.success) {
      console.log(`   ✅ SUCCESS - Ready to save`);
    } else {
      console.log(`   ⚠️  NEEDS ATTENTION`);
      overallValidation.issues.forEach((issue) =>
        console.log(`      ❌ ${issue}`)
      );
    }

    // Detailed subject list log
    console.log(`\n📋 EXTRACTED SUBJECTS:`);
    console.log("=".repeat(90));

    const sortedSubjects = [...uniqueSubjects].sort((a, b) => {
      const yearOrder = {
        "1st Year": 1,
        "2nd Year": 2,
        "3rd Year": 3,
        "4th Year": 4,
      };
      const semOrder = { "1st Semester": 1, "2nd Semester": 2, Summer: 3 };

      if (yearOrder[a.yearLevel] !== yearOrder[b.yearLevel]) {
        return yearOrder[a.yearLevel] - yearOrder[b.yearLevel];
      }
      return semOrder[a.semester] - semOrder[b.semester];
    });

    let currentYear = "";
    sortedSubjects.forEach((s, i) => {
      if (s.yearLevel !== currentYear) {
        currentYear = s.yearLevel;
        console.log(`\n${currentYear}:`);
      }
      const sem = s.semester
        .replace(" Semester", " Sem")
        .replace("Summer", "Sum");
      console.log(
        `  ${String(i + 1).padStart(2)}. ${s.subjectCode.padEnd(18)} | ${s.subjectName.substring(0, 45).padEnd(45)} | ${s.units}u | ${sem}`
      );
    });

    console.log(`\n${"=".repeat(90)}`);

    // Missing subjects log
    if (overallValidation.missingCourses.length > 0) {
      console.log(
        `\n⚠️  MISSING SUBJECTS (${overallValidation.missingCourses.length}):`
      );
      overallValidation.missingCourses.forEach((code) => {
        const ref = CURRICULUM_REFERENCE[code];
        if (ref) {
          console.log(
            `   ❌ ${code.padEnd(18)} - ${ref.name} (${ref.yearLevel}, ${ref.semester})`
          );
        } else {
          console.log(`   ❌ ${code}`);
        }
      });
    }

    console.log("=".repeat(90));

    // Response
    res.json({
      success: overallValidation.success,
      quality: overallValidation.quality,
      data: {
        subjects: uniqueSubjects,
        totalSubjectsFound: uniqueSubjects.length,
        imagesProcessed: files.length,
        imageCount: files.length,
      },
      validation: overallValidation,
      imageResults: imageResults,
      imageSummary: {
        image1Good: imageResults[0]?.validation?.isGood || false,
        image2Good: imageResults[1]?.validation?.isGood || false,
        totalImages: files.length,
      },
    });
  } catch (error) {
    console.error("❌ Fatal error:", error.message);
    res.status(error.status || 500).json({
      success: false,
      error: getErrorMessage(error),
      quality: "error",
    });
  } finally {
    for (const file of files) {
      try {
        fs.unlinkSync(file.path);
      } catch {}
    }
  }
});

// ==================== SCAN GRADES ENDPOINT ====================
app.post("/api/scan-grades", upload.single("image"), async (req, res) => {
  console.log("\n📊 Received grade scan request");

  if (!req.file) {
    return res.status(400).json({ success: false, error: "No image uploaded" });
  }

  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({
      success: false,
      error: "GROQ_API_KEY not set. Add it to your .env file.",
    });
  }

  const file = req.file;
  console.log(
    `📎 Image: ${file.originalname} (${(file.size / 1024).toFixed(1)}KB)`,
  );
  console.log("🤖 Sending to Groq (Llama 4 Scout) for grade extraction...");

  try {
    const parsed = await sendToGroq(file, GRADES_PROMPT);

    const grades = (parsed.grades || [])
      .map((g, idx) => ({
        subjectCode: g.subjectCode || `UNKNOWN_${idx + 1}`,
        subjectName: g.subjectName || "",
        grade: parseFloat(g.grade) || 0,
      }))
      .filter((g) => g.grade > 0 && g.grade <= 5.0);

    console.log(`✅ Extracted ${grades.length} grades`);
    grades.forEach((g) => console.log(`   📊 ${g.subjectCode}: ${g.grade}`));

    res.json({
      success: true,
      data: {
        grades,
        totalFound: grades.length,
        confidence: parsed.confidence || "unknown",
      },
    });
  } catch (error) {
    console.error("❌ Grade scan error:", error.message);
    res.status(error.status || 500).json({
      success: false,
      error: getErrorMessage(error),
    });
  } finally {
    try {
      fs.unlinkSync(file.path);
    } catch {}
  }
});

// ==================== SCAN TIMETABLE ENDPOINT ====================
app.post("/api/scan-timetable", upload.single("image"), async (req, res) => {
  console.log("\n📅 Received timetable scan request");

  if (!req.file) {
    return res.status(400).json({ success: false, error: "No image uploaded" });
  }

  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({
      success: false,
      error: "GROQ_API_KEY not set. Add it to your .env file.",
    });
  }

  const file = req.file;
  console.log(
    `📎 Image: ${file.originalname} (${(file.size / 1024).toFixed(1)}KB)`,
  );
  console.log("🤖 Sending to Groq (Llama 4 Scout) for timetable extraction...");

  try {
    const parsed = await sendToGroq(file, TIMETABLE_PROMPT);

    const academicYear = parsed.academicYear || "";
    const semester = parsed.semester || "";

    const subjects = (parsed.subjects || []).map((s, idx) => ({
      subjectName: s.subjectName || `Unknown Subject ${idx + 1}`,
      subjectCode: s.subjectCode || "",
      section: s.section || "",
      room: s.room || "",
      instructor: s.instructor || "",
      schedules: (s.schedules || []).map((sched) => ({
        day: sched.day || "",
        startTime: sched.startTime || "",
        endTime: sched.endTime || "",
      })),
    }));

    console.log(`✅ Extracted ${subjects.length} subjects from timetable`);
    console.log(`   📆 ${academicYear} - ${semester}`);
    subjects.forEach((s) => {
      const days = s.schedules
        .map((sc) => `${sc.day} ${sc.startTime}-${sc.endTime}`)
        .join(", ");
      console.log(`   📚 ${s.subjectName} | ${s.instructor} | ${days}`);
    });

    res.json({
      success: true,
      data: {
        academicYear,
        semester,
        subjects,
        totalSubjectsFound: subjects.length,
        confidence: parsed.confidence || "unknown",
      },
    });
  } catch (error) {
    console.error("❌ Timetable scan error:", error.message);
    res.status(error.status || 500).json({
      success: false,
      error: getErrorMessage(error),
    });
  } finally {
    try {
      fs.unlinkSync(file.path);
    } catch {}
  }
});

// ==================== HEALTH CHECK ====================
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    provider: "Groq (Llama 4 Scout)",
    apiKeySet: !!process.env.GROQ_API_KEY,
    features: ["image_preprocessing", "quality_validation", "auto_hydration"],
  });
});

// ==================== START ====================
app.listen(PORT, "0.0.0.0", () => {
  console.log("");
  console.log("╔════════════════════════════════════════════╗");
  console.log("║   🚀 ClasSync Groq Server (ENHANCED)      ║");
  console.log("╠════════════════════════════════════════════╣");
  console.log(`║   URL:   http://localhost:${PORT}              ║`);
  console.log(`║   Model: Llama 4 Scout 17B                ║`);
  console.log(
    `║   Key:   ${process.env.GROQ_API_KEY ? "✅ Set" : "❌ Missing"}                           ║`,
  );
  console.log("║   Features: Image Enhancement + QC        ║");
  console.log("╠════════════════════════════════════════════╣");
  console.log("║   POST /api/scan-cor         (COR scan)   ║");
  console.log("║   POST /api/scan-curriculum  (curriculum)  ║");
  console.log("║   POST /api/scan-grades      (grades)      ║");
  console.log("║   POST /api/scan-timetable   (timetable)   ║");
  console.log("║   GET  /api/health           (status)      ║");
  console.log("╚════════════════════════════════════════════╝");

  if (!process.env.GROQ_API_KEY) {
    console.log("");
    console.log("⚠️  Create .env file with:");
    console.log("   GROQ_API_KEY=gsk_your_key_here");
    console.log("");
    console.log("   Get your free key: https://console.groq.com/keys");
  }

  console.log("");
});