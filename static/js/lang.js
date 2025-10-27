function applyLang(lang) {
    const root = document.getElementById("html-root");
    if (!root) return;

    if (lang === "ar") {
        root.setAttribute("lang","ar");
        root.setAttribute("dir","rtl");
        document.body.classList.add("rtl");
    } else {
        root.setAttribute("lang","en");
        root.setAttribute("dir","ltr");
        document.body.classList.remove("rtl");
    }

    document.querySelectorAll(".lang-en").forEach(el=>{
        el.style.display = (lang === "en") ? "inline" : "none";
    });
    document.querySelectorAll(".lang-ar").forEach(el=>{
        el.style.display = (lang === "ar") ? "inline" : "none";
    });

    localStorage.setItem("growsmart-lang", lang);
}

function toggleLang() {
    const current = localStorage.getItem("growsmart-lang") || "en";
    const next = current === "en" ? "ar" : "en";
    applyLang(next);
}

// On load, restore lang
(function() {
    const saved = localStorage.getItem("growsmart-lang") || "en";
    applyLang(saved);
})();
