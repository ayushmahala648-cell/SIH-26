const drawer = document.getElementById("drawer");
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const orClick = document.getElementById("orClick");
const previewImage = document.getElementById("previewImage");
const uploadText = document.getElementById("uploadText");
const selectBtn = document.getElementById("select-btn");

let selectedSampleFilename = null;
let uploadedFilePath = null;

function openDrawer() {
    drawer.classList.add("open");
}

function closeDrawer() {
    drawer.classList.remove("open");
}

if (orClick && fileInput) {
    orClick.addEventListener("click", function () {
        fileInput.click();
    });
}

if (uploadArea && fileInput) {
    uploadArea.addEventListener("dragover", function (e) {
        e.preventDefault();
        uploadArea.style.borderColor = "#05055d";
    });

    uploadArea.addEventListener("dragleave", function (e) {
        e.preventDefault();
        uploadArea.style.borderColor = "rgb(14, 178, 210)";
    });

    uploadArea.addEventListener("drop", function (e) {
        e.preventDefault();
        uploadArea.style.borderColor = "rgb(14, 178, 210)";
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

if (fileInput) {
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    showImage(file);
    selectBtn.textContent = "Uploading...";
    selectBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (data.success) {
            uploadedFilePath = data.image_path;
            selectedSampleFilename = null;
            sessionStorage.setItem("paridhi_image_type", "uploaded");
            sessionStorage.setItem("paridhi_image_path", data.image_path);
            sessionStorage.setItem("paridhi_image_name", data.filename);

            selectBtn.disabled = false;
            selectBtn.textContent = "Start AI Analysis";
            setupStartAnalysisButton(`new_analysis.html?type=uploaded&path=${encodeURIComponent(data.image_path)}&name=${encodeURIComponent(data.filename)}`);
        } else {
            alert("Upload failed: " + (data.detail || "Unknown error"));
            selectBtn.textContent = "Select Image";
            selectBtn.disabled = false;
        }
    } catch (err) {
        console.error("Upload error:", err);
        // Fallback for local preview if server offline
        sessionStorage.setItem("paridhi_image_type", "local");
        selectBtn.disabled = false;
        selectBtn.textContent = "Start AI Analysis";
        setupStartAnalysisButton("new_analysis.html?sample=aerialPhoto1.jpg.jpeg");
    }
}

const sampleImages = document.querySelectorAll(".sampleImage");

sampleImages.forEach(function (image) {
    image.addEventListener("click", function () {
        const fullSrc = image.src;
        const filename = fullSrc.substring(fullSrc.lastIndexOf("/") + 1);

        selectedSampleFilename = filename;
        uploadedFilePath = null;
        sessionStorage.setItem("paridhi_image_type", "sample");
        sessionStorage.setItem("paridhi_image_name", filename);

        uploadArea.innerHTML = `<img src="${image.src}" class="uploadedImage" style="width:100%;height:100%;object-fit:contain;border-radius:15px;">`;
        closeDrawer();

        selectBtn.textContent = "Start AI Analysis";
        setupStartAnalysisButton(`new_analysis.html?type=sample&name=${encodeURIComponent(filename)}`);

        let changeImage = document.getElementById("changeImage");
        if (!changeImage) {
            changeImage = document.createElement("button");
            changeImage.textContent = "Change Image";
            changeImage.id = "changeImage";
            selectBtn.parentElement.appendChild(changeImage);

            changeImage.addEventListener("click", function () {
                openDrawer();
                changeImage.classList.add("active");
                selectBtn.classList.add("notActive");
            });
        }
    });
});

function setupStartAnalysisButton(targetUrl) {
    selectBtn.onclick = () => {
        selectBtn.classList.add("active");
        let changeImage = document.getElementById("changeImage");
        if (changeImage) changeImage.classList.remove("active");
        window.location.href = targetUrl;
    };
}

function showImage(file) {
    const reader = new FileReader();
    reader.onload = function (event) {
        uploadArea.innerHTML = `
            <img src="${event.target.result}" class="uploadedImage" style="width:100%;height:100%;object-fit:contain;border-radius:15px;">
        `;
    };
    reader.readAsDataURL(file);
}