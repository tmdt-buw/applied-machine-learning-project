document.addEventListener('DOMContentLoaded', function() {
    // Load header component first
    const headerContainer = document.querySelector('[data-component="pages/components/header.html"]');
    
    // Function to load a component
    function loadComponent(container) {
        const componentPath = container.getAttribute('data-component');
        
        return fetch(componentPath)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to load component: ${componentPath}`);
                }
                return response.text();
            })
            .then(html => {
                container.innerHTML = html;
                
                // Execute any scripts that were in the loaded HTML
                container.querySelectorAll('script').forEach(oldScript => {
                    const newScript = document.createElement('script');
                    Array.from(oldScript.attributes).forEach(attr => {
                        newScript.setAttribute(attr.name, attr.value);
                    });
                    newScript.textContent = oldScript.textContent;
                    oldScript.parentNode.replaceChild(newScript, oldScript);
                });
            })
            .catch(error => {
                console.error(error);
                container.innerHTML = `<div class="alert alert-danger">Error loading component: ${error.message}</div>`;
            });
    }
    
    // First load header, then load all other components
    if (headerContainer) {
        loadComponent(headerContainer).then(() => {
            // After header is loaded, load all other components
            document.querySelectorAll('[data-component]:not([data-component="pages/components/header.html"])').forEach(container => {
                loadComponent(container);
            });
        });
    } else {
        // If no header, load all components
        document.querySelectorAll('[data-component]').forEach(container => {
            loadComponent(container);
        });
    }
}); 