document.addEventListener('DOMContentLoaded', function() {
    // 模擬從資料庫來的資料
    const dummyData = Array.from({ length: 123 }, (_, i) => ({
        id: i + 1,
        name: `使用者${i + 1}`,
        email: `user${i + 1}@example.com`
    }));

    let currentPage = 1;
    let itemsPerPage = parseInt(document.getElementById('items-per-page').value, 10);

    const tableBody = document.getElementById('data-table-body');
    const itemsPerPageSelect = document.getElementById('items-per-page');
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const pageNumbersContainer = document.getElementById('page-numbers');

    function displayData() {
        tableBody.innerHTML = '';
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const paginatedData = dummyData.slice(startIndex, endIndex);

        paginatedData.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.id}</td>
                <td>${item.name}</td>
                <td>${item.email}</td>
                <td>
                    <button class="edit" onclick="alert('編輯 ID: ${item.id}')">編輯</button>
                    <button class="delete" onclick="confirm('確定要刪除 ID: ${item.id}?')">刪除</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    function updatePaginationControls() {
        const totalPages = Math.ceil(dummyData.length / itemsPerPage);
        pageNumbersContainer.innerHTML = `第 ${currentPage} / ${totalPages} 頁`;

        prevPageButton.disabled = currentPage === 1;
        nextPageButton.disabled = currentPage === totalPages;
    }

    function goToPage(page) {
        currentPage = page;
        displayData();
        updatePaginationControls();
    }

    itemsPerPageSelect.addEventListener('change', (e) => {
        itemsPerPage = parseInt(e.target.value, 10);
        goToPage(1); // Reset to first page
    });

    prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            goToPage(currentPage - 1);
        }
    });

    nextPageButton.addEventListener('click', () => {
        const totalPages = Math.ceil(dummyData.length / itemsPerPage);
        if (currentPage < totalPages) {
            goToPage(currentPage + 1);
        }
    });

    // 初始化
    goToPage(1);
});
