const drawer = document.getElementById("drawer");
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const orClick = document.getElementById("orClick");
const previewImage = document.getElementById("previewImage");
const uploadText = document.getElementById("uploadText");




function openFilePicker() {
    fileInput.click();
}




function openDrawer() {
    drawer.classList.add("open")
}

function closeDrawer() {
    drawer.classList.remove("open");
}


orClick.addEventListener("click", function () {

    fileInput.click();

});



fileInput.addEventListener("change", function () {

    if (fileInput.files.length > 0) {

        const file = fileInput.files[0];

        showImage(file);

    }

});


const sampleImages = document.querySelectorAll(".sampleImage");

sampleImages.forEach(function (image) {

    image.addEventListener("click", function () {

        uploadArea.innerHTML = `
    <img src="${image.src}" class="uploadedImage">
`;
        closeDrawer();

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