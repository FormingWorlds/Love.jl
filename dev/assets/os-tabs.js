(function () {
  function activate(group, key) {
    document.querySelectorAll(`.os-tabs[data-tab-group="${group}"]`).forEach((tabs) => {
      tabs.querySelectorAll('[role="tab"]').forEach((btn) => {
        btn.setAttribute("aria-selected", String(btn.dataset.tab === key));
      });
      tabs.querySelectorAll('[role="tabpanel"]').forEach((panel) => {
        panel.dataset.active = String(panel.dataset.tab === key);
      });
    });
    try {
      localStorage.setItem(`os-tab:${group}`, key);
    } catch (e) {
      /* localStorage unavailable (private browsing etc.) — selection just won't persist */
    }
  }

  function init() {
    document.querySelectorAll(".os-tabs").forEach((tabs) => {
      const group = tabs.dataset.tabGroup;
      tabs.querySelectorAll('[role="tab"]').forEach((btn) => {
        btn.addEventListener("click", () => activate(group, btn.dataset.tab));
      });
    });
    const groups = new Set(Array.from(document.querySelectorAll(".os-tabs")).map((t) => t.dataset.tabGroup));
    groups.forEach((group) => {
      let saved = null;
      try { saved = localStorage.getItem(`os-tab:${group}`); } catch (e) { /* ignore */ }
      if (saved) activate(group, saved);
    });
  }

  document.readyState === "loading"
    ? document.addEventListener("DOMContentLoaded", init)
    : init();
})();