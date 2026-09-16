const drawer = document.getElementById("drawer");
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const orClick = document.getElementById("orClick");
const previewImage = document.getElementById("previewImage");
const uploadText = document.getElementById("uploadText");
const selectBtn = document.getElementById("select-btn");

let selectedFile = null;


function openDrawer() {
    if (drawer) {
        drawer.classList.add("open");
    }
}


function closeDrawer() {
    if (drawer) {
        drawer.classList.remove("open");
    }
}


// "or click to browse"
if (orClick && fileInput) {

    orClick.addEventListener("click", function () {

        fileInput.click();

    });

}


// FILE SELECTED FROM COMPUTER
if (fileInput && selectBtn) {

    fileInput.addEventListener("change", function () {

        if (fileInput.files.length > 0) {

            selectedFile = fileInput.files[0];

            showImage(selectedFile);

            selectBtn.textContent = "Start Analysis";

        }

    });

}


// SELECT IMAGE / START ANALYSIS BUTTON
if (selectBtn && fileInput) {

    selectBtn.addEventListener("click", function () {

        if (selectedFile === null) {

            fileInput.click();

        } else {

            uploadImage();

        }

    });

}


// SAMPLE IMAGES
const sampleImages = document.querySelectorAll(".sampleImage");

sampleImages.forEach(function (image) {

    image.addEventListener("click", async function () {

        // Show selected sample image
        uploadArea.innerHTML = `
            <img src="${image.src}" class="uploadedImage">
        `;

        closeDrawer();


        // Convert sample image into a File
        try {

            const response = await fetch(image.src);
            const blob = await response.blob();

            selectedFile = new File(
                [blob],
                "sample-image.jpeg",
                { type: blob.type }
            );

            selectBtn.textContent = "Start Analysis";


        } catch (error) {

            console.error("Could not load sample image:", error);

        }


        // CREATE CHANGE IMAGE BUTTON
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


function showImage(file) {

    const reader = new FileReader();

    reader.onload = function (event) {

        uploadArea.innerHTML = `
            <img src="${event.target.result}" class="uploadedImage">
        `;

    };

    reader.readAsDataURL(file);

}


async function uploadImage() {

    if (!selectedFile) {

        alert("Please select an image first.");
        return;

    }


    const formData = new FormData();

    // IMPORTANT: "file" must match FastAPI's UploadFile parameter
    formData.append("file", selectedFile);


    try {

        const response = await fetch(
            "http://localhost:8000/api/upload",
            {
                method: "POST",
                body: formData
            }
        );


        if (!response.ok) {

            const errorText = await response.text();

            console.error(
                "Backend error:",
                response.status,
                errorText
            );

            throw new Error(
                `Upload failed: ${response.status}`
            );

        }


        const data = await response.json();

        console.log("Backend response:", data);


        // Go to New Analysis ONLY after successful upload
        window.location.href = "new_analysis.html";


    } catch (error) {

        console.error("Upload error:", error);

        alert("Something went wrong while uploading.");

    }

}