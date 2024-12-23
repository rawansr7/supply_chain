
# class UploadInventoryData(View):
#     def post(self, request):
#         file = request.FILES['file']
#         inventory_data = InventoryData(uploaded_file=file)
#         inventory_data.save()

#         # Preprocess data here
#         df = pd.read_csv(inventory_data.uploaded_file.path)
#         # Example preprocessing: Drop irrelevant columns
#         df.drop(columns=['snap_CA', 'snap_TX', 'snap_WI'], errors='ignore', inplace=True)
#         df['date'] = pd.to_datetime(df['date'])

#         # Save preprocessed data (can be extended to a DB or file storage)
#         df.to_csv(f"preprocessed_{inventory_data.id}.csv", index=False)

#         return JsonResponse({'message': 'File uploaded and preprocessed successfully!'})


# def upload_page(request):
#     return render(request, 'upload.html')
