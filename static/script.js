document.addEventListener('DOMContentLoaded', function(){
    const uploadForm = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const scannedImage = document.getElementById('scannedImage');

    uploadForm.addEventListener('submit', function(e){
        e.preventDefault();
        const formData = new FormData(uploadForm);

        fetch('/scan', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if(data.error){
                alert("Error: " + data.error);
            } else {
                scannedImage.src = data.scanned_image_url;
                resultDiv.style.display = 'block';
            }
        })
        .catch(err => {
            console.error(err);
            alert("An error occurred during document scanning.");
        });
    });
});
