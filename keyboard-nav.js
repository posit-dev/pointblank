/**
 * Keyboard Navigation & Shortcuts for Great Docs
 *
 * Features:
 * - `/` or `s` to focus search
 * - `[` / `]` for previous/next page navigation
 * - `q` to navigate to homepage
 * - `m` or `n` to show/hide floating menu overlay (sidebar nav or navbar links)
 * - `d` to toggle dark mode
 * - `c` to copy page as Markdown (when available)
 * - `h` or `?` to show/hide keyboard shortcuts help overlay
 * - `Escape` to close overlay or unfocus active element
 * - Dark mode aware
 * - Respects reduced-motion preferences
 * - Skips shortcuts when user is typing in inputs/textareas
 * - ARIA attributes for screen reader accessibility
 */

(function() {
    'use strict';

    // i18n helper — read translations from <meta name="gd-i18n">
    var _i18nCache = null;
    function _gdT(key, fallback) {
        if (!_i18nCache) {
            try {
                var meta = document.querySelector('meta[name="gd-i18n"]');
                _i18nCache = meta ? JSON.parse(meta.getAttribute('content')) : {};
            } catch (e) { _i18nCache = {}; }
        }
        return _i18nCache[key] || fallback;
    }

    // State
    var overlay = null;
    var isOverlayOpen = false;
    var menuOverlay = null;
    var isMenuOpen = false;

    /**
     * Check if the user prefers reduced motion
     */
    function prefersReducedMotion() {
        return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    /**
     * Check if the active element is an input or editable area
     */
    function isTyping() {
        var el = document.activeElement;
        if (!el) return false;
        var tag = el.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
        if (el.isContentEditable) return true;
        // Quarto search autocomplete input
        if (el.classList.contains('aa-Input')) return true;
        return false;
    }

    // ── Search Focus ─────────────────────────────────────────────────────

    function focusSearch() {
        // Try Quarto's Algolia-style autocomplete input first
        var searchInput = document.querySelector('.aa-Input');
        if (!searchInput) {
            // Fall back to the Quarto search container (clicking it usually opens the search)
            var searchContainer = document.querySelector('#quarto-search');
            if (searchContainer) {
                var btn = searchContainer.querySelector('button') ||
                          searchContainer.querySelector('[role="button"]');
                if (btn) {
                    btn.click();
                    return;
                }
            }
        }
        if (searchInput) {
            searchInput.focus();
            if (searchInput.select) searchInput.select();
        }
    }

    // ── Page Navigation ──────────────────────────────────────────────────

    function navigatePrev() {
        var link = document.querySelector('.nav-page-previous a.pagination-link');
        if (link) {
            link.click();
        }
    }

    function navigateNext() {
        var link = document.querySelector('.nav-page-next a.pagination-link');
        if (link) {
            link.click();
        }
    }

    function navigateHome() {
        // Use the navbar brand link (always points to site root)
        var brand = document.querySelector('.navbar-brand');
        if (brand && brand.href) {
            window.location.href = brand.href;
            return;
        }
        // Fallback: navigate to site root
        window.location.href = './';
    }

    // ── Show Menu Overlay ──────────────────────────────────────────────

    /**
     * Detect whether the current page is the homepage.
     * Homepage body has `nav-fixed` but NOT `nav-sidebar`.
     */
    function isHomepage() {
        return !document.body.classList.contains('nav-sidebar');
    }

    /**
     * Build menu items from the top navbar links (used on homepage).
     * Returns an array of {text, href, active} objects.
     */
    function getNavbarItems() {
        var items = [];
        var links = document.querySelectorAll(
            '#navbarCollapse > .navbar-nav.me-auto > .nav-item > .nav-link'
        );
        for (var i = 0; i < links.length; i++) {
            var a = links[i];
            var text = a.querySelector('.menu-text');
            if (!text) continue;
            items.push({
                text: text.textContent.trim(),
                href: a.getAttribute('href') || '#',
                active: a.classList.contains('active'),
                section: null
            });
        }
        return items;
    }

    /**
     * Build menu items from the left sidebar navigation (used on content pages).
     * Handles both sectioned sidebars (User Guide) and flat sidebars (Recipes).
     * Returns an array of {text, href, active, section} objects.
     */
    function getSidebarItems() {
        var items = [];
        var sidebar = document.getElementById('quarto-sidebar');
        if (!sidebar) return items;

        var sections = sidebar.querySelectorAll('.sidebar-item-section');

        if (sections.length > 0) {
            // Sectioned sidebar (e.g., User Guide pages)
            for (var s = 0; s < sections.length; s++) {
                var sec = sections[s];
                var headerEl = sec.querySelector(':scope > .sidebar-item-container .menu-text');
                var sectionTitle = headerEl ? headerEl.textContent.trim() : null;

                var links = sec.querySelectorAll('.sidebar-section .sidebar-link');
                for (var i = 0; i < links.length; i++) {
                    var a = links[i];
                    var text = a.querySelector('.menu-text');
                    if (!text) continue;
                    items.push({
                        text: text.textContent.trim(),
                        href: a.getAttribute('href') || '#',
                        active: a.classList.contains('active'),
                        section: sectionTitle
                    });
                    sectionTitle = null;
                }
            }
        } else {
            // Flat sidebar (e.g., Recipes, custom sections)
            var links = sidebar.querySelectorAll('.sidebar-item .sidebar-link');
            for (var i = 0; i < links.length; i++) {
                var a = links[i];
                var text = a.querySelector('.menu-text');
                if (!text) continue;
                items.push({
                    text: text.textContent.trim(),
                    href: a.getAttribute('href') || '#',
                    active: a.classList.contains('active'),
                    section: null
                });
            }
        }

        return items;
    }

    /**
     * Create the floating menu overlay element.
     */
    function createMenuOverlay() {
        var el = document.createElement('div');
        el.className = 'gd-menu-overlay';
        el.setAttribute('role', 'dialog');
        el.setAttribute('aria-modal', 'true');
        el.setAttribute('aria-label',
            _gdT('kb_show_menu', 'Show menu'));

        var homepage = isHomepage();
        var items = homepage ? getNavbarItems() : getSidebarItems();
        var title = homepage
            ? _gdT('kb_menu_title_site', 'Site Navigation')
            : _gdT('kb_menu_title_section', 'Section Navigation');

        var html = '<div class="gd-menu-overlay-inner">';
        html += '<div class="gd-menu-overlay-header">';
        html += '<h2>' + escapeHtml(title) + '</h2>';
        html += '<button class="gd-menu-overlay-close" aria-label="' +
                escapeHtml(_gdT('kb_close_dismiss', 'Close')) +
                '" type="button">';
        html += '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" ' +
                'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
                'stroke-linejoin="round" aria-hidden="true">' +
                '<line x1="18" y1="6" x2="6" y2="18"></line>' +
                '<line x1="6" y1="6" x2="18" y2="18"></line>' +
                '</svg>';
        html += '</button>';
        html += '</div>';

        html += '<nav class="gd-menu-overlay-body" aria-label="' + escapeHtml(title) + '">';
        html += '<ul class="gd-menu-list">';

        var currentSection = null;
        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            // Section divider
            if (item.section && item.section !== currentSection) {
                currentSection = item.section;
                html += '<li class="gd-menu-section" aria-hidden="true">' +
                        escapeHtml(currentSection) + '</li>';
            }
            var activeClass = item.active ? ' gd-menu-item-active' : '';
            html += '<li><a class="gd-menu-item' + activeClass +
                    '" href="' + escapeHtml(item.href) + '">' +
                    escapeHtml(item.text) + '</a></li>';
        }

        html += '</ul>';
        html += '</nav></div>';

        el.innerHTML = html;
        return el;
    }

    // Menu keyboard navigation state
    var menuSelectedIndex = -1;

    /**
     * Get all navigable menu items from the overlay.
     */
    function getMenuItems() {
        if (!menuOverlay) return [];
        return menuOverlay.querySelectorAll('.gd-menu-item');
    }

    /**
     * Update the visual focus indicator on menu items.
     */
    function updateMenuFocus() {
        var items = getMenuItems();
        for (var i = 0; i < items.length; i++) {
            if (i === menuSelectedIndex) {
                items[i].classList.add('gd-menu-item-focused');
                items[i].scrollIntoView({ block: 'nearest' });
            } else {
                items[i].classList.remove('gd-menu-item-focused');
            }
        }
    }

    /**
     * Handle keyboard events within the menu overlay.
     */
    function handleMenuKeydown(e) {
        var items = getMenuItems();
        if (items.length === 0) return;

        switch (e.key) {
            case 'ArrowDown':
            case 'ArrowRight':
            case 'ArrowUp':
            case 'ArrowLeft':
                e.preventDefault();
                // Any arrow press selects the first item
                if (menuSelectedIndex < 0) {
                    menuSelectedIndex = 0;
                } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                    menuSelectedIndex = Math.min(menuSelectedIndex + 1, items.length - 1);
                } else {
                    menuSelectedIndex = Math.max(menuSelectedIndex - 1, 0);
                }
                updateMenuFocus();
                break;

            case 'Enter':
                if (menuSelectedIndex >= 0 && menuSelectedIndex < items.length) {
                    e.preventDefault();
                    items[menuSelectedIndex].click();
                }
                break;
        }
    }

    function showMenu() {
        // Always recreate to pick up current page state
        if (menuOverlay) {
            menuOverlay.remove();
            menuOverlay = null;
        }

        menuOverlay = createMenuOverlay();
        document.body.appendChild(menuOverlay);
        menuSelectedIndex = -1;

        // Close button
        var closeBtn = menuOverlay.querySelector('.gd-menu-overlay-close');
        closeBtn.addEventListener('click', hideMenu);

        // Click on backdrop
        menuOverlay.addEventListener('click', function(e) {
            if (e.target === menuOverlay) {
                hideMenu();
            }
        });

        // Force reflow then show
        void menuOverlay.offsetWidth;
        menuOverlay.classList.add('gd-menu-overlay-visible');
        isMenuOpen = true;
    }

    function hideMenu() {
        if (!menuOverlay) return;
        menuOverlay.classList.remove('gd-menu-overlay-visible');
        isMenuOpen = false;
        menuSelectedIndex = -1;
        // Remove after transition
        setTimeout(function() {
            if (menuOverlay && !isMenuOpen) {
                menuOverlay.remove();
                menuOverlay = null;
            }
        }, 200);
    }

    // ── Copy Page ────────────────────────────────────────────────────────

    function copyPage() {
        var btn = document.querySelector('.gd-copy-md-btn');
        if (btn) btn.click();
    }

    // ── Dark Mode Toggle ─────────────────────────────────────────────────

    function toggleDarkMode() {
        // Delegate to the existing dark-mode-toggle button if present
        var btn = document.getElementById('dark-mode-toggle');
        if (btn) {
            btn.click();
            return;
        }
    }

    // ── Help Overlay ─────────────────────────────────────────────────────

    var SHORTCUT_GROUPS = [
        {
            titleKey: 'kb_group_navigation',
            titleFallback: 'Navigation',
            shortcuts: [
                { keys: ['s', '/'], descKey: 'kb_focus_search', descFallback: 'Focus search' },
                { keys: ['['], descKey: 'kb_prev_page', descFallback: 'Previous page' },
                { keys: [']'], descKey: 'kb_next_page', descFallback: 'Next page' },
                { keys: ['m', 'n'], descKey: 'kb_show_menu', descFallback: 'Show menu' },
                { keys: ['q'], descKey: 'kb_home', descFallback: 'Go to homepage' },
            ]
        },
        {
            titleKey: 'kb_group_display',
            titleFallback: 'Display',
            shortcuts: [
                { keys: ['d'], descKey: 'kb_toggle_dark', descFallback: 'Toggle dark mode' },
                { keys: ['c'], descKey: 'kb_copy_page', descFallback: 'Copy page as Markdown' },
            ]
        },
        {
            titleKey: 'kb_group_general',
            titleFallback: 'General',
            shortcuts: [
                { keys: ['h', '?'], descKey: 'kb_show_help', descFallback: 'Show keyboard shortcuts' },
                { keys: ['Esc'], descKey: 'kb_close_dismiss', descFallback: 'Close / dismiss' },
            ]
        },
    ];

    function createOverlay() {
        var el = document.createElement('div');
        el.className = 'gd-keyboard-overlay';
        el.setAttribute('role', 'dialog');
        el.setAttribute('aria-modal', 'true');
        el.setAttribute('aria-label',
            _gdT('kb_overlay_title', 'Keyboard shortcuts'));

        var title = _gdT('kb_overlay_title', 'Keyboard shortcuts');

        var html = '<div class="gd-keyboard-overlay-inner">';
        html += '<div class="gd-keyboard-overlay-header">';
        html += '<h2>' + escapeHtml(title) + '</h2>';
        html += '<button class="gd-keyboard-overlay-close" aria-label="' +
                escapeHtml(_gdT('kb_close_dismiss', 'Close')) +
                '" type="button">';
        html += '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" ' +
                'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
                'stroke-linejoin="round" aria-hidden="true">' +
                '<line x1="18" y1="6" x2="6" y2="18"></line>' +
                '<line x1="6" y1="6" x2="18" y2="18"></line>' +
                '</svg>';
        html += '</button>';
        html += '</div>';

        html += '<div class="gd-keyboard-overlay-body">';

        for (var g = 0; g < SHORTCUT_GROUPS.length; g++) {
            var group = SHORTCUT_GROUPS[g];
            html += '<div class="gd-keyboard-group">';
            html += '<h3>' + escapeHtml(_gdT(group.titleKey, group.titleFallback)) + '</h3>';
            html += '<dl>';
            for (var s = 0; s < group.shortcuts.length; s++) {
                var sc = group.shortcuts[s];
                html += '<div class="gd-keyboard-row">';
                html += '<dt>';
                for (var k = 0; k < sc.keys.length; k++) {
                    if (k > 0) html += ' <span class="gd-keyboard-or">' +
                        escapeHtml(_gdT('kb_or', 'or')) + '</span> ';
                    html += '<kbd>' + escapeHtml(sc.keys[k]) + '</kbd>';
                }
                html += '</dt>';
                html += '<dd>' + escapeHtml(_gdT(sc.descKey, sc.descFallback)) + '</dd>';
                html += '</div>';
            }
            html += '</dl>';
            html += '</div>';
        }

        html += '</div></div>';

        el.innerHTML = html;
        return el;
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    function showOverlay() {
        if (!overlay) {
            overlay = createOverlay();
            document.body.appendChild(overlay);

            // Close button
            var closeBtn = overlay.querySelector('.gd-keyboard-overlay-close');
            closeBtn.addEventListener('click', hideOverlay);

            // Click on backdrop
            overlay.addEventListener('click', function(e) {
                if (e.target === overlay) {
                    hideOverlay();
                }
            });
        }

        if (prefersReducedMotion()) {
            overlay.classList.add('gd-keyboard-overlay-visible');
        } else {
            // Force reflow before adding the class for animation
            void overlay.offsetWidth;
            overlay.classList.add('gd-keyboard-overlay-visible');
        }
        isOverlayOpen = true;

        // Focus the close button for keyboard users
        var closeBtn = overlay.querySelector('.gd-keyboard-overlay-close');
        if (closeBtn) closeBtn.focus();
    }

    function hideOverlay() {
        if (!overlay) return;
        overlay.classList.remove('gd-keyboard-overlay-visible');
        isOverlayOpen = false;
    }

    // ── Keyboard Event Handler ───────────────────────────────────────────

    function handleKeydown(e) {
        // When menu is open, handle all menu-related keys
        if (isMenuOpen) {
            if (e.key === 'Escape' || e.key === 'm' || e.key === 'n') {
                e.preventDefault();
                hideMenu();
                return;
            }
            // Delegate arrow keys and Enter to menu navigation
            handleMenuKeydown(e);
            return;
        }

        // When help overlay is open, handle Escape and toggle keys (h/?)
        if (isOverlayOpen) {
            if (e.key === 'Escape' || e.key === 'h' || e.key === '?') {
                e.preventDefault();
                hideOverlay();
            }
            return;
        }

        // Skip if user is typing in an input
        if (isTyping()) return;

        // Skip if modifier keys are held (allow browser/OS shortcuts)
        if (e.ctrlKey || e.metaKey || e.altKey) return;

        switch (e.key) {
            case '?':
            case 'h':
                e.preventDefault();
                showOverlay();
                break;

            case 's':
                e.preventDefault();
                focusSearch();
                break;

            // Note: '/' is handled by sidebar-filter.js when it's active.
            // When sidebar-filter is not present, we handle it here.
            case '/':
                // Only if sidebar-filter hasn't already handled it
                if (!document.querySelector('.sidebar-filter-input')) {
                    e.preventDefault();
                    focusSearch();
                }
                break;

            case '[':
                e.preventDefault();
                navigatePrev();
                break;

            case ']':
                e.preventDefault();
                navigateNext();
                break;

            case 'm':
            case 'n':
                e.preventDefault();
                showMenu();
                break;

            case 'q':
                e.preventDefault();
                navigateHome();
                break;

            case 'd':
                e.preventDefault();
                toggleDarkMode();
                break;

            case 'c':
                e.preventDefault();
                copyPage();
                break;

            case 'Escape':
                // Blur the active element
                if (document.activeElement && document.activeElement !== document.body) {
                    document.activeElement.blur();
                }
                break;
        }
    }

    // ── Navbar Button ────────────────────────────────────────────────

    /**
     * Create the keyboard shortcuts navbar button.
     * Uses the Lucide "keyboard" icon (MIT-licensed).
     */
    function createKeyboardButton() {
        var container = document.createElement('div');
        container.id = 'gd-keyboard-btn-container';

        var btn = document.createElement('button');
        btn.id = 'gd-keyboard-btn';
        btn.className = 'gd-keyboard-btn';
        btn.type = 'button';
        btn.setAttribute('aria-label',
            _gdT('kb_show_help', 'Show keyboard shortcuts'));
        btn.setAttribute('data-tippy-content',
            _gdT('kb_show_help', 'Show keyboard shortcuts'));
        btn.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" ' +
            'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
            '<path d="M10 8h.01"/><path d="M12 12h.01"/>' +
            '<path d="M14 8h.01"/><path d="M16 12h.01"/>' +
            '<path d="M18 8h.01"/><path d="M6 8h.01"/>' +
            '<path d="M7 16h10"/><path d="M8 12h.01"/>' +
            '<rect width="20" height="16" x="2" y="4" rx="2"/>' +
            '</svg>';

        container.appendChild(btn);
        return container;
    }

    /**
     * Insert the keyboard button into the navbar.
     */
    function insertKeyboardButton() {
        // Insert into navbarCollapse; navbar-widgets.js will collect it
        // into the unified #gd-navbar-widgets container.
        var collapse = document.getElementById('navbarCollapse');
        if (!collapse) return;

        var container = createKeyboardButton();
        var navItem = document.createElement('li');
        navItem.className = 'nav-item';
        navItem.appendChild(container);

        // Append to navbarCollapse (the collector will move it)
        collapse.appendChild(navItem);

        // Click handler
        var btn = document.getElementById('gd-keyboard-btn');
        if (btn) {
            btn.addEventListener('click', function() {
                if (isOverlayOpen) {
                    hideOverlay();
                } else {
                    showOverlay();
                }
            });

            // Initialize tooltip directly — window.tippy is available from
            // Quarto's bundle before any include-after-body scripts run.
            if (window.tippy) {
                window.tippy(btn, {
                    content: btn.getAttribute('data-tippy-content'),
                    placement: 'bottom',
                    animation: 'shift-away',
                    duration: [200, 150],
                    delay: [0, 0],
                    arrow: true,
                });
            }
        }
    }

    // ── Initialization ───────────────────────────────────────────────────

    function init() {
        document.addEventListener('keydown', handleKeydown);
        insertKeyboardButton();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
