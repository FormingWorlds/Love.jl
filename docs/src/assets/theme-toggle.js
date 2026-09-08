document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("proteus-theme-toggle")) return;

  const navRight =
    document.querySelector(".docs-right-buttons") ||
    document.querySelector("header.docs-navbar .docs-right") ||
    document.querySelector("header.docs-navbar");

  const picker = document.getElementById("documenter-themepicker");

  if (!navRight || !picker) {
    console.warn("PROTEUS theme toggle: navbar or theme picker not found");
    return;
  }

  // We only cycle between these two; "auto" and catppuccin variants
  // are still reachable via the settings gear if someone wants them.
  const isDark = (name) => name === "documenter-dark";

  const setTheme = (name) => {
    picker.value = name;
    // Dispatch the same event Documenter's own picker listener expects,
    // so its existing (already correct) swap logic runs — we don't
    // touch <link> or <html> ourselves at all.
    picker.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const toggleBtn = document.createElement("a");
  toggleBtn.id = "proteus-theme-toggle";
  toggleBtn.className = "docs-theme-toggle nav-item";
  toggleBtn.title = "Toggle light/dark theme";
  toggleBtn.style.cursor = "pointer";
  toggleBtn.style.padding = "0 0.5rem";
  toggleBtn.style.display = "inline-flex";
  toggleBtn.style.alignItems = "center";

  const sunIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`;
  const moonIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

  const updateIcon = () => {
    toggleBtn.innerHTML = isDark(picker.value) ? sunIcon : moonIcon;
  };

  updateIcon();

  toggleBtn.addEventListener("click", (e) => {
    e.preventDefault();
    setTheme(isDark(picker.value) ? "documenter-light" : "documenter-dark");
    updateIcon();
  });

  // Keep the icon in sync if the theme changes via some other path
  // (e.g. the settings gear, if it's still reachable, or OS preference
  // changes under "Automatic").
  picker.addEventListener("change", updateIcon);

  navRight.insertBefore(toggleBtn, navRight.firstChild);
});