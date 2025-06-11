

document.getElementById("profile-form").addEventListener("submit", function(event) {
    event.preventDefault();

    const username = document.getElementById("username").value.trim();
    const bio = document.getElementById("bio").value.trim();

    if (!username || !bio) {
        alert("Please fill in all required fields.");
        return;
    }

    document.getElementById("result-section").style.display = "block";
    document.getElementById("prediction-output").innerText = "Analyzing... (This is a demo output)";

    // Simulated delay for analysis
    setTimeout(() => {
        document.getElementById("prediction-output").innerText = 
            `Prediction: This profile appears to be REAL ✅ (confidence: 92%)`;
    }, 1500);
});
