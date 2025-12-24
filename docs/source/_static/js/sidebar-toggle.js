/**
 * Sidebar Navigation Toggle
 *
 * Makes entire navigation rows clickable to expand/collapse their children.
 * Works with pydata-sphinx-theme's <details>/<summary> structure.
 */
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.querySelector('.bd-sidebar-primary');
    if (!sidebar) return;

    // Find all nav items that have a details element (expandable)
    const expandableItems = sidebar.querySelectorAll('li.has-children');

    expandableItems.forEach(function(navItem) {
        const navLink = navItem.querySelector(':scope > a');
        const details = navItem.querySelector(':scope > details');

        if (!navLink || !details) return;

        // Handle click on the nav link
        navLink.addEventListener('click', function(e) {
            // Toggle the details open/closed
            if (details.open) {
                // Currently open - close it
                details.open = false;
                e.preventDefault();
                e.stopPropagation();
            } else {
                // Currently closed - open it and navigate
                details.open = true;
                // Allow navigation to proceed
            }
        });
    });

    // Expand only level 1 by default (shows categories like Algebraic, BBOB, etc.)
    expandableItems.forEach(function(navItem) {
        const details = navItem.querySelector(':scope > details');
        if (!details) return;

        // Only expand L1 items (Test Functions, etc.)
        const isL1 = navItem.classList.contains('toctree-l1');

        if (isL1) {
            details.open = true;
        }
    });
});
