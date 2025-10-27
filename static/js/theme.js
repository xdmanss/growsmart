function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("growsmart-theme", theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    const next = current === "light" ? "dark" : "light";
    applyTheme(next);
}

// On load, restore theme
(function() {
    const saved = localStorage.getItem("growsmart-theme");
    if (saved === "dark" || saved === "light") {
        applyTheme(saved);
    } else {
        applyTheme("light");
    }
})();
