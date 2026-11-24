// The homepage's showcase: the stepper that builds a validation one part at a time, the switch
// between the ways of looking at what failed, and the copy button
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // Copy text, and show that it's copied on the button (its icon, for a moment)
  function copyText(button, text) {
    const done = () => {
      const icon = button.querySelector(".bi");
      button.classList.add("pbh-copied");
      if (icon) icon.className = "bi bi-check2";
      setTimeout(() => {
        button.classList.remove("pbh-copied");
        if (icon) icon.className = "bi bi-clipboard";
      }, 1400);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, () => {});
    } else {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
        done();
      } catch (_) {}
      area.remove();
    }
  }

  function setUpCopyButtons() {
    for (const button of document.querySelectorAll("[data-copy]")) {
      button.addEventListener("click", () => copyText(button, button.dataset.copy));
    }
  }

  // The lines of a code cell's code, as elements, each with the line break after it (so that a line
  // can be folded away; the code's text, which the copy button copies, is as it was)
  function codeLines(cell) {
    const lines = Array.from(cell.querySelectorAll("pre code > span"));
    for (const line of lines) {
      const after = line.nextSibling;
      if (after && after.nodeType === Node.TEXT_NODE && after.data === "\n") line.appendChild(after);
    }
    return lines;
  }

  // Fold the runs of lines that aren't new (but the line before and after each new one) into a
  // line that says how many there are, which unfolds them when clicked
  function foldUnchanged(lines, added) {
    const keep = lines.map((_, k) => added[k] || added[k - 1] || added[k + 1]);
    let k = 0;
    while (k < lines.length) {
      if (keep[k]) {
        k++;
        continue;
      }
      let end = k;
      while (end < lines.length && !keep[end]) end++;
      const run = lines.slice(k, end);
      if (run.length >= 3) {
        const [mark, ...rest] = run;
        mark.classList.add("pbh-fold-mark");
        mark.dataset.fold = `${run.length} unchanged lines`;
        mark.tabIndex = 0;
        mark.setAttribute("role", "button");
        mark.setAttribute("aria-label", `Show ${run.length} unchanged lines`);
        for (const line of rest) line.classList.add("pbh-folded");
        const unfold = () => {
          mark.classList.remove("pbh-fold-mark");
          mark.removeAttribute("role");
          mark.removeAttribute("tabindex");
          mark.removeAttribute("aria-label");
          for (const line of rest) line.classList.remove("pbh-folded");
        };
        mark.addEventListener("click", unfold);
        mark.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            unfold();
          }
        });
      }
      k = end;
    }
  }

  // Which lines of `next` aren't in `prev` (by the longest common subsequence of the two)
  function newLines(prev, next) {
    const m = prev.length;
    const n = next.length;
    const lcs = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        lcs[i][j] = prev[i] === next[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
      }
    }
    const added = new Array(n).fill(true);
    let i = 0;
    let j = 0;
    while (i < m && j < n) {
      if (prev[i] === next[j]) {
        added[j] = false;
        i++;
        j++;
      } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
        i++;
      } else {
        j++;
      }
    }
    return added;
  }

  // Panels, one shown at a time, chosen by tabs: numbered ones over the stepper (which marks each
  // step's new lines of code, shows the step's table before its code, and plays through the steps
  // while it's in view), and chips over the switch
  function setUpPanels(container, { panel, name, stepper, label }) {
    const panels = Array.from(container.querySelectorAll(`:scope > .${panel}`));
    if (panels.length === 0) return;

    const tabs = document.createElement("div");
    tabs.className = stepper ? "pbh-tabs" : "pbh-chips";
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", label);

    const viewport = document.createElement("div");
    viewport.className = "pbh-panels";
    const stack = document.createElement("div");
    stack.className = "pbh-stack";
    viewport.appendChild(stack);

    const id = container.id || (stepper ? "pbh-stepper" : "pbh-switch");
    let previous = [];

    const buttons = panels.map((step, index) => {
      // The panel's name (its first paragraph) goes on its tab
      const nameEl = step.querySelector(`.${name}`);
      const title = nameEl ? nameEl.textContent.trim() : `Step ${index + 1}`;
      if (nameEl) {
        const para = nameEl.closest("p");
        (para && para.parentElement === step ? para : nameEl).remove();
      }

      const button = document.createElement("button");
      button.type = "button";
      button.id = `${id}-tab-${index}`;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", `${id}-panel-${index}`);
      if (stepper) {
        button.className = "pbh-tab";
        button.innerHTML = `<span class="pbh-tab-num">${index + 1}</span><span class="pbh-tab-name"></span><span class="pbh-tab-progress" aria-hidden="true"></span>`;
        button.querySelector(".pbh-tab-name").textContent = title;
      } else {
        button.className = "pbh-chip";
        button.textContent = title;
      }
      tabs.appendChild(button);

      step.id = `${id}-panel-${index}`;
      step.setAttribute("role", "tabpanel");
      step.setAttribute("aria-labelledby", button.id);
      step.classList.add("pbh-panel");

      // The code's lines, with those new to this step marked
      const cell = step.querySelector(".cell");
      if (stepper && cell) {
        const lines = codeLines(cell);
        const text = lines.map((line) => line.textContent.replace(/\s+$/, ""));
        const added = newLines(previous, text);
        let count = 0;
        lines.forEach((line, k) => {
          if (added[k] && text[k] !== "" && previous.length > 0) {
            line.classList.add("pbh-new");
            count++;
          }
        });
        if (previous.length > 0) foldUnchanged(lines, added);
        previous = text;
        const code = cell.querySelector(".cell-code");
        if (code && count > 0) {
          const badge = document.createElement("span");
          badge.className = "pbh-new-count";
          badge.textContent = `${count} new line${count === 1 ? "" : "s"}`;
          code.appendChild(badge);
        }
        // The table first, then the code that made it
        const output = cell.querySelector(".cell-output-display");
        if (code && output) cell.insertBefore(output, code);
      }

      stack.appendChild(step);
      return button;
    });

    container.prepend(tabs);
    container.appendChild(viewport);

    let active = -1;
    let observer = null;

    function fitHeight() {
      if (active < 0) return;
      viewport.style.height = `${panels[active].offsetHeight}px`;
    }

    function scrollToNew(step) {
      const pre = step.querySelector(".cell-code pre");
      const first = step.querySelector(".pbh-new");
      if (!pre || !first) return;
      const top = first.offsetTop - pre.offsetTop - 12;
      pre.scrollTo({ top: Math.max(0, top), behavior: reduceMotion.matches ? "auto" : "smooth" });
    }

    function show(index, focus = false) {
      if (index === active) return;
      active = index;
      panels.forEach((step, k) => {
        const on = k === index;
        step.classList.toggle("pbh-active", on);
        step.inert = !on;
        step.setAttribute("aria-hidden", on ? "false" : "true");
        buttons[k].setAttribute("aria-selected", on ? "true" : "false");
        buttons[k].tabIndex = on ? 0 : -1;
        if (stepper) buttons[k].classList.toggle("pbh-done", k < index);
      });
      if (focus) buttons[index].focus();
      if (observer) {
        observer.disconnect();
        observer.observe(panels[index]);
      }
      fitHeight();
      if (stepper) scrollToNew(panels[index]);
    }

    // Autoplay (for the stepper): through the steps while it's in view, until the reader takes over
    const DWELL = 5200;
    let timer = null;
    let playing = stepper && !reduceMotion.matches;
    let visible = false;

    function stopPlaying() {
      playing = false;
      container.classList.remove("pbh-playing");
      clearTimeout(timer);
      timer = null;
    }

    function schedule() {
      clearTimeout(timer);
      timer = null;
      if (!playing || !visible) {
        container.classList.remove("pbh-playing");
        return;
      }
      if (active >= panels.length - 1) {
        stopPlaying();
        return;
      }
      // (Restarting the tab's progress bar)
      container.classList.remove("pbh-playing");
      void container.offsetWidth;
      container.style.setProperty("--pbh-dwell", `${DWELL}ms`);
      container.classList.add("pbh-playing");
      timer = setTimeout(() => {
        show(active + 1);
        schedule();
      }, DWELL);
    }

    buttons.forEach((button, index) => {
      button.addEventListener("click", () => {
        stopPlaying();
        show(index);
      });
    });

    tabs.addEventListener("keydown", (event) => {
      const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
      let next = null;
      if (event.key in keys) next = (active + keys[event.key] + panels.length) % panels.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = panels.length - 1;
      if (next === null) return;
      event.preventDefault();
      stopPlaying();
      show(next, true);
    });

    // Working with the table (or its code) stops the autoplay too
    viewport.addEventListener("pointerdown", stopPlaying);
    viewport.addEventListener("focusin", stopPlaying);

    if ("ResizeObserver" in window) {
      observer = new ResizeObserver(fitHeight);
    }
    window.addEventListener("resize", fitHeight);

    container.classList.add("pbh-ready");
    show(0);

    if (stepper && "IntersectionObserver" in window) {
      new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            visible = entry.isIntersecting;
            schedule();
          }
        },
        // (In view: across the middle of the window, however tall the stepper is)
        { rootMargin: "-35% 0px -35% 0px" },
      ).observe(container);
    }
  }

  function init() {
    setUpCopyButtons();
    document.querySelectorAll(".pbh-stepper").forEach((stepper) =>
      setUpPanels(stepper, {
        panel: "pbh-step",
        name: "pbh-step-name",
        stepper: true,
        label: "The parts of a validation",
      }),
    );
    document.querySelectorAll(".pbh-switch").forEach((switcher) =>
      setUpPanels(switcher, {
        panel: "pbh-option",
        name: "pbh-option-name",
        stepper: false,
        label: "Ways to look at what failed",
      }),
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
