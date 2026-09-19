const drawer = document.getElementById("drawer");
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const orClick = document.getElementById("orClick");
const previewImage = document.getElementById("previewImage");
const uploadText = document.getElementById("uploadText");
const selectBtn = document.getElementById("select-btn");
const plusBtn = document.getElementById("plus");

let currentfile = null;


const API_BASE = window.location.origin && window.location.origin.includes(":8000")
    ? ""
    : "http://localhost:8000";

function openDrawer() {
    if (drawer) drawer.classList.add("open");
}

function closeDrawer() {
    if (drawer) drawer.classList.remove("open");
}


if (orClick && fileInput) {
    orClick.addEventListener("click", function () {
        fileInput.click();
    });
}

if (plusBtn) {
    plusBtn.addEventListener("click", function () {
        openDrawer();
    });
}


if (fileInput) {
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            currentfile = file;

            showImagePreview(file);
            Startanalysis();
        }
    });
}


const sampleImages = document.querySelectorAll(".sampleImage");
sampleImages.forEach((image) => {
    image.addEventListener("click", async () => {
        
        if (uploadArea) {
            uploadArea.innerHTML = `<img src="${image.src}" class="uploadedImage">`;
        }
        closeDrawer();
        Startanalysis();

        
        const filename = image.src.split("/").pop() || "sample_aerial.jpg";
        try {
            const response = await fetch(image.src);
            const blob = await response.blob();
            currentfile = new File([blob], filename, { type: blob.type || "image/jpeg" });
        } catch (err) {
            console.warn("fetch() on local file failed, using canvas fallback:", err);
            
            try {
                const canvas = document.createElement("canvas");
                canvas.width = image.naturalWidth || 800;
                canvas.height = image.naturalHeight || 600;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(image, 0, 0);
                canvas.toBlob((blob) => {
                    if (blob) {
                        currentfile = new File([blob], filename, { type: "image/jpeg" });
                    }
                }, "image/jpeg", 0.95);
            } catch (canvasErr) {
                console.error("Canvas export failed:", canvasErr);
            }
        }
    });
});


function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function (event) {
        if (uploadArea) {
            uploadArea.innerHTML = `<img src="${event.target.result}" class="uploadedImage">`;
        }
    };
    reader.readAsDataURL(file);
}

function Startanalysis() {
    if (!selectBtn) return;
    selectBtn.textContent = "Start Analysis";
    selectBtn.onclick = handleUploadAndAnalyze;

    let changeImage = document.getElementById("changeImage");
    if (!changeImage) {
        changeImage = document.createElement("button");
        changeImage.textContent = "Change Image";
        changeImage.id = "changeImage";
        if (selectBtn.parentElement) {
            selectBtn.parentElement.appendChild(changeImage);
        }

        changeImage.addEventListener("click", () => {
            openDrawer();
            changeImage.classList.add("active");
            selectBtn.classList.add("notActive");
        });
    }
}


async function handleUploadAndAnalyze() {
    if (!currentfile) {
        alert("Please select or upload an image first!");
        return;
    }

    selectBtn.textContent = "Uploading to server...";
    selectBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", currentfile);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (response.ok && data.success) {
            localStorage.setItem("uploadedImageFilename", data.filename);
            localStorage.setItem("uploadedImageUrl", `${API_BASE}${data.image_url}`);
            console.log("Uploaded successfully:", data);

            window.location.href = "new_analysis.html";
        } else {
            alert(`Upload failed: ${data.detail || "Server error"}`);
            selectBtn.textContent = "Start Analysis";
            selectBtn.disabled = false;
        }
    } catch (err) {
        console.error("Backend connection error:", err);
        alert("Could not connect to FastAPI server. Make sure it is running on http://localhost:8000!");
        selectBtn.textContent = "Start Analysis";
        selectBtn.disabled = false;
    }
}
