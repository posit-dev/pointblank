/**
 * Skill Switcher — sticky tab bar for multi-skill pages.
 *
 * Renders as a pill-style switcher that sticks below the navbar when the user
 * scrolls past it. Each pill corresponds to a skill panel; clicking a pill
 * hides the current panel and reveals the selected one with no page reload.
 */
(function () {
  "use strict";

  var bar = document.querySelector(".gd-skill-switcher");
  if (!bar) return;

  var pills = bar.querySelectorAll(".gd-skill-pill");
  var panels = document.querySelectorAll(".gd-skill-panel");
  if (!pills.length || !panels.length) return;

  // ── Pill click handler ──
  function activate(idx) {
    pills.forEach(function (p, i) {
      p.classList.toggle("gd-skill-pill--active", i === idx);
      p.setAttribute("aria-selected", i === idx ? "true" : "false");
      p.setAttribute("tabindex", i === idx ? "0" : "-1");
    });
    panels.forEach(function (panel, i) {
      panel.hidden = i !== idx;
    });
  }

  pills.forEach(function (pill, i) {
    pill.addEventListener("click", function () {
      activate(i);
    });

    // Keyboard navigation (left/right arrows within the switcher)
    pill.addEventListener("keydown", function (e) {
      var dir = 0;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") dir = 1;
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") dir = -1;
      if (e.key === "Home") { activate(0); pills[0].focus(); e.preventDefault(); return; }
      if (e.key === "End") { var last = pills.length - 1; activate(last); pills[last].focus(); e.preventDefault(); return; }
      if (!dir) return;
      e.preventDefault();
      var next = (i + dir + pills.length) % pills.length;
      activate(next);
      pills[next].focus();
    });
  });

  // ── Sticky behaviour ──
  // Compute a sentinel position: when the bar scrolls past the navbar, stick it.
  var sentinel = document.createElement("div");
  sentinel.className = "gd-skill-switcher-sentinel";
  bar.parentNode.insertBefore(sentinel, bar);

  /**
   * Compute the visible height of the navbar area above the page content.
   *
   * On mobile, headroom.js hides the header via `transform: translateY(-100%)`
   * while the `.quarto-secondary-nav` counter-translates to stay visible. In
   * that state the full `#quarto-header.offsetHeight` is wrong because the
   * element is off-screen; the effective top is just the secondary nav height.
   */
  function visibleNavHeight() {
    var header = document.getElementById("quarto-header") || document.querySelector("nav.navbar");
    if (!header) return 0;

    // On mobile (<992px) when headroom has unpinned the header, only the
    // secondary nav is visible. On desktop the CSS keeps the navbar pinned
    // via translateY(0%) !important, so ignore the class there.
    if (window.innerWidth < 992 && header.classList.contains("headroom--unpinned")) {
      var secNav = header.querySelector(".quarto-secondary-nav");
      return secNav ? secNav.offsetHeight : 0;
    }

    return header.offsetHeight;
  }

  function updateSticky() {
    var navH = visibleNavHeight();
    var rect = sentinel.getBoundingClientRect();
    if (rect.top <= navH) {
      bar.classList.add("gd-skill-switcher--stuck");
      bar.style.top = navH + "px";
      // Reserve space so content doesn't jump
      sentinel.style.height = bar.offsetHeight + "px";
    } else {
      bar.classList.remove("gd-skill-switcher--stuck");
      bar.style.top = "";
      sentinel.style.height = "0";
    }
  }

  window.addEventListener("scroll", updateSticky, { passive: true });
  window.addEventListener("resize", updateSticky, { passive: true });
  // Re-evaluate when headroom pins/unpins the navbar
  window.addEventListener("quarto-hrChanged", updateSticky);
  updateSticky();

  // Activate the first pill on load
  activate(0);
})();
