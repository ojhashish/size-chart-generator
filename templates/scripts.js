document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('sizePredictorForm');
    form.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent the default form submission

        // Validate form data
        if (validateForm()) {
            displayResult(); // Display results in the result div
        }
    });
});

function validateForm() {
    var gender = document.getElementById("Gender").value.trim();
    var height = document.getElementById("Height").value;
    var cupSize = document.getElementById("Cup Size").value.trim().toUpperCase();

    // Capitalize the first letter of gender
    var genderFormatted = gender.charAt(0).toUpperCase() + gender.slice(1).toLowerCase();

    // Validate gender
    if (genderFormatted !== "Male" && genderFormatted !== "Female") {
        alert("Please enter 'Male' or 'Female' for gender.");
        return false;
    }

    // Validate height format (e.g., 5'7")
    var heightRegex = /^\d{1,2}'\d{1,2}"$/;
    if (!heightRegex.test(height)) {
        alert("Height must be in the format e.g., 5'7\"");
        return false;
    }

    // Validate cup size if gender is Female
    var cupSizeRequired = genderFormatted === "Female";
    var cupSizeRegex = /^[A-Z]$/;
    if (cupSizeRequired && !cupSizeRegex.test(cupSize)) {
        alert("If gender is Female, Cup Size must be a single character (A-Z).");
        return false;
    }

    // Check if Cup Size field is empty when required
    if (cupSizeRequired && !cupSize) {
        alert("Cup Size is required for Female.");
        return false;
    }

    // Set formatted values back to the fields
    document.getElementById("Gender").value = genderFormatted;
    document.getElementById("Cup Size").value = cupSize;

    return true;
}

function displayResult() {
    // Example result for demonstration purposes
    var resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "<h2>Predicted Size</h2><p>Your predicted size is: M (Medium)</p>";
    resultDiv.style.display = "block";
}
