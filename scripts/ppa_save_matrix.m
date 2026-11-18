function ppa_save_matrix(path, A)
% PPA_SAVE_MATRIX  Write a dense matrix as space-separated ASCII rows.
  fid = fopen(path, "w");
  if fid < 0
    error("cannot write %s", path);
  end
  for i = 1:size(A, 1)
    for j = 1:size(A, 2)
      if j > 1
        fprintf(fid, " ");
      end
      fprintf(fid, "%.17g", A(i, j));
    end
    fprintf(fid, "\n");
  end
  fclose(fid);
end
