document.addEventListener('DOMContentLoaded', function() {
    // Handle search type selection to show/hide relationship fields
    const searchTypeSelect = document.getElementById('search_type');
    const relationshipFields = document.getElementById('relationship-fields');

    searchTypeSelect.addEventListener('change', function() {
        if (this.value === 'relationship') {
            relationshipFields.style.display = 'block';
        } else {
            relationshipFields.style.display = 'none';
        }
    });

    // Handle search form submission
    const searchForm = document.getElementById('search-form');
    const resultsContainer = document.getElementById('results-container');

    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        resultsContainer.innerHTML = `
            <div class="d-flex justify-content-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
        
        // Collect form data
        const formData = new FormData(searchForm);
        
        // Send search request
        fetch('/search', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultsContainer.innerHTML = `
                    <div class="alert alert-danger">
                        Error: ${data.error}
                    </div>
                `;
                return;
            }
            
            displaySearchResults(data.results, formData.get('search_type'));
        })
        .catch(error => {
            resultsContainer.innerHTML = `
                <div class="alert alert-danger">
                    An error occurred: ${error.message}
                </div>
            `;
        });
    });

    // Handle document links
    const documentLinks = document.querySelectorAll('.document-link');
    const documentModal = new bootstrap.Modal(document.getElementById('documentModal'));
    const documentDetails = document.getElementById('document-details');
    const documentModalLabel = document.getElementById('documentModalLabel');

    documentLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const filename = this.dataset.filename;
            documentModalLabel.textContent = `Document: ${filename}`;
            
            // Show loading indicator
            documentDetails.innerHTML = `
                <div class="d-flex justify-content-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
            
            // Open the modal
            documentModal.show();
            
            // Fetch document details
            fetch(`/document/${filename}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        documentDetails.innerHTML = `
                            <div class="alert alert-danger">
                                Error: ${data.error}
                            </div>
                        `;
                        return;
                    }
                    
                    displayDocumentDetails(data, filename);
                })
                .catch(error => {
                    documentDetails.innerHTML = `
                        <div class="alert alert-danger">
                            An error occurred: ${error.message}
                        </div>
                    `;
                });
        });
    });

    // Function to display search results
    function displaySearchResults(results, searchType) {
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="alert alert-info">
                    No results found for your search.
                </div>
            `;
            return;
        }

        let resultsHTML = '';

        if (searchType === 'entity') {
            resultsHTML = `
                <div class="mb-3">
                    <h6>Found ${results.length} entities</h6>
                </div>
                <div class="result-list">
            `;

            results.forEach(result => {
                const similarity = 100 - (result.distance || 0);
                
                resultsHTML += `
                    <div class="result-item">
                        <h5>${result.text || 'Unnamed Entity'}</h5>
                        <div class="result-meta">
                            <span class="badge bg-primary">${result.type || 'unknown'}</span>
                            <span class="ms-2">Source: ${result.source_file || 'unknown'}</span>
                            <span class="ms-2 confidence-score">${similarity.toFixed(2)}% match</span>
                        </div>
                    </div>
                `;
            });

            resultsHTML += '</div>';
        } else if (searchType === 'statement') {
            resultsHTML = `
                <div class="mb-3">
                    <h6>Found ${results.length} statements</h6>
                </div>
                <div class="result-list">
            `;

            results.forEach(result => {
                const similarity = 100 - (result.distance || 0);
                
                resultsHTML += `
                    <div class="result-item">
                        <div class="result-text">"${result.statement || 'Unknown statement'}"</div>
                        <div class="result-meta mt-2">
                            <span class="badge bg-primary">${result.type || 'unknown'}</span>
                            <span class="ms-2">Source: ${result.source_file || 'unknown'}</span>
                            <span class="ms-2 confidence-score">${similarity.toFixed(2)}% match</span>
                        </div>
                    </div>
                `;
            });

            resultsHTML += '</div>';
        } else if (searchType === 'relationship') {
            resultsHTML = `
                <div class="mb-3">
                    <h6>Found ${results.length} relationships</h6>
                </div>
                <div class="result-list">
            `;

            results.forEach(result => {
                const similarity = 100 - (result.distance || 0);
                
                resultsHTML += `
                    <div class="result-item">
                        <div class="relationship">
                            <span class="relationship-subject">${result.subject || 'Unknown'}</span>
                            <span class="relationship-relation">${result.relation || 'related to'}</span>
                            <span class="relationship-object">${result.object || 'Unknown'}</span>
                        </div>
                        <div class="result-meta mt-2">
                            <span class="badge bg-secondary">From sentence</span>
                            <span class="confidence-score ms-2">${similarity.toFixed(2)}% match</span>
                        </div>
                        <div class="result-text mt-2">
                            "${result.sentence || 'No context available'}"
                        </div>
                        <div class="text-end mt-2">
                            <small class="text-muted">Source: ${result.source_file || 'unknown'}</small>
                        </div>
                    </div>
                `;
            });

            resultsHTML += '</div>';
        }

        resultsContainer.innerHTML = resultsHTML;
    }

    // Function to display document details
    function displayDocumentDetails(data, filename) {
        // Create tabs for different sections
        let tabsHtml = `
            <ul class="nav nav-tabs doc-tabs mb-3" id="documentTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab">Overview</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="entities-tab" data-bs-toggle="tab" data-bs-target="#entities" type="button" role="tab">Entities</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="relationships-tab" data-bs-toggle="tab" data-bs-target="#relationships" type="button" role="tab">Relationships</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="statements-tab" data-bs-toggle="tab" data-bs-target="#statements" type="button" role="tab">Statements</button>
                </li>
            </ul>
        `;

        // Overview tab content
        let overviewHtml = `
            <div class="document-section">
                <h5 class="document-section-title">Document Information</h5>
                <p><strong>Filename:</strong> ${filename}</p>
                <p><strong>Entities:</strong> ${Object.keys(data.entities || {}).length}</p>
                <p><strong>Relationships:</strong> ${(data.relationships || []).length}</p>
                <p><strong>Statements:</strong> ${(data.statements || []).length}</p>
                <p><strong>Events:</strong> ${(data.events || []).length}</p>
            </div>
        `;

        // Entities tab content
        let entitiesHtml = '<div class="row">';
        
        // Group entities by type
        const entityTypes = {};
        
        for (const [entityId, entity] of Object.entries(data.entities || {})) {
            const type = entity.type || 'unknown';
            if (!entityTypes[type]) {
                entityTypes[type] = [];
            }
            entityTypes[type].push({id: entityId, ...entity});
        }
        
        for (const [type, entitiesList] of Object.entries(entityTypes)) {
            entitiesHtml += `
                <div class="col-md-6 mb-3">
                    <div class="entity-card">
                        <div class="entity-header">${type.toUpperCase()} (${entitiesList.length})</div>
                        <div class="entity-body">
                            <ul class="entity-list">
            `;
            
            entitiesList.forEach(entity => {
                entitiesHtml += `<li>${entity.text}</li>`;
            });
            
            entitiesHtml += `
                            </ul>
                        </div>
                    </div>
                </div>
            `;
        }
        
        entitiesHtml += '</div>';

        // Relationships tab content
        let relationshipsHtml